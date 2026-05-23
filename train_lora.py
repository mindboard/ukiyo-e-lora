"""
SD 1.5 LoRA 学習スクリプト（VRAM 8GB 向け）
被写体保持型のスタイル LoRA。dataset/imgs と dataset/txt から学習。
"""
import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer

from peft import LoraConfig, get_peft_model
#from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

from peft.utils import get_peft_model_state_dict
from diffusers.utils import convert_state_dict_to_diffusers
from peft.utils import get_peft_model_state_dict as peft_state_dict

import os
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"


# -------------------- Dataset --------------------
class PairedDataset(Dataset):
    """imgs/<stem>.png と txt/<stem>.txt をペアで読み込む"""

    def __init__(self, root: Path, tokenizer, resolution: int = 512):
        self.imgs_dir = root / "imgs"
        self.txt_dir = root / "txt"
        self.tokenizer = tokenizer

        # 拡張子を問わず png/jpg を拾う
        self.stems = sorted({
            p.stem for p in self.imgs_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        })
        # キャプションが存在するものだけに絞る
        self.stems = [s for s in self.stems if (self.txt_dir / f"{s}.txt").exists()]
        if not self.stems:
            raise RuntimeError("ペアが見つからない。imgs/ と txt/ を確認")

        self.tx = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(resolution),
            transforms.RandomHorizontalFlip(p=0.5),  # 浮世絵は左右反転OKな構図が多い
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # [-1, 1] へ
        ])
        print(f"loaded {len(self.stems)} pairs from {root}")

    def __len__(self):
        return len(self.stems)

    def _find_image(self, stem: str) -> Path:
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            p = self.imgs_dir / f"{stem}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(stem)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        img = Image.open(self._find_image(stem)).convert("RGB")
        pixel = self.tx(img)

        caption = (self.txt_dir / f"{stem}.txt").read_text(encoding="utf-8").strip()
        ids = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        return {"pixel_values": pixel, "input_ids": ids}


# -------------------- main --------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="dataset")
    p.add_argument("--output_dir", type=str, default="output/lora_ukiyoe")
    p.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=4)  # 実効 batch 4
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    p.add_argument("--use_8bit_adam", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    accel = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision=args.mixed_precision,
    )

    # --- モデルをロード ---
    tokenizer = CLIPTokenizer.from_pretrained(args.base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.base_model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.base_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.base_model, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(args.base_model, subfolder="scheduler")

    # text_encoder と vae は凍結
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    # --- UNet に LoRA を挿す ---
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
        bias="none",
    )
    unet = get_peft_model(unet, lora_cfg)
    unet.print_trainable_parameters()

    # --- VRAM 節約 ---
    #unet.enable_gradient_checkpointing()

    try:
        unet.enable_xformers_memory_efficient_attention()
    except Exception:
        pass  # xformers 未インストールでもOK（PyTorch 2.x なら sdpa が効く）

    # --- 推論時の dtype（VAE/text_enc は fp16 で凍結のまま GPU へ）---
    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32
    vae.to(accel.device, dtype=weight_dtype)
    text_encoder.to(accel.device, dtype=weight_dtype)
    # unet は LoRA 部分が学習対象なので fp32 のまま（accel が mixed precision で扱う）

    # --- Optimizer ---
    trainable = [p for p in unet.parameters() if p.requires_grad]
    if args.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(trainable, lr=args.lr)
    else:
        optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    # --- DataLoader ---
    dataset = PairedDataset(Path(args.data_root), tokenizer, resolution=args.resolution)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # --- LR Scheduler ---
    lr_sched = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=args.max_steps * args.grad_accum,
    )

    unet, optimizer, loader, lr_sched = accel.prepare(unet, optimizer, loader, lr_sched)

    # --- 学習ループ ---
    global_step = 0
    epochs_needed = math.ceil(args.max_steps / math.ceil(len(loader) / args.grad_accum))
    progress = tqdm(total=args.max_steps, disable=not accel.is_main_process)

    for epoch in range(epochs_needed):
        unet.train()
        for batch in loader:
            with accel.accumulate(unet):
                # 画像 → latent
                with torch.no_grad():
                    latents = vae.encode(
                        batch["pixel_values"].to(dtype=weight_dtype)
                    ).latent_dist.sample() * vae.config.scaling_factor

                # ノイズと timestep を sample
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,),
                    device=latents.device,
                ).long()
                noisy = noise_scheduler.add_noise(latents, noise, timesteps)

                # テキスト埋め込み
                with torch.no_grad():
                    enc_hidden = text_encoder(batch["input_ids"])[0]

                # ターゲット決定（v_prediction の場合の分岐）
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(noise_scheduler.config.prediction_type)

                pred = unet(noisy, timesteps, enc_hidden).sample
                loss = F.mse_loss(pred.float(), target.float(), reduction="mean")

                accel.backward(loss)
                if accel.sync_gradients:
                    accel.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                lr_sched.step()
                optimizer.zero_grad()

            if accel.sync_gradients:
                global_step += 1
                progress.update(1)
                progress.set_postfix(loss=loss.detach().item())

                # 中間保存
                if global_step % args.save_every == 0 and accel.is_main_process:
                    save_path = out / f"step-{global_step}"
                    save_lora(accel.unwrap_model(unet), save_path)

            if global_step >= args.max_steps:
                break
        if global_step >= args.max_steps:
            break

    # --- 最終保存 ---
    if accel.is_main_process:
        save_lora(accel.unwrap_model(unet), out / "final")
    accel.end_training()


#def save_lora(unet_peft, save_dir: Path):
#    """diffusers が読める形式で LoRA 重みだけ保存"""
#    save_dir.mkdir(parents=True, exist_ok=True)
#    state_dict = peft_state_dict(unet_peft)
#    # diffusers の load_lora_weights が読めるキー名に変換
#    state_dict = {f"unet.{k}": v for k, v in state_dict.items()}
#    StableDiffusionPipeline.save_lora_weights(
#        save_directory=str(save_dir),
#        unet_lora_layers=state_dict,
#        safe_serialization=True,
#    )
#    print(f"saved LoRA to {save_dir}")
#

#from peft.utils import get_peft_model_state_dict
#from diffusers.utils import convert_state_dict_to_diffusers

#def save_lora(unet_peft, save_dir: Path):
#    save_dir.mkdir(parents=True, exist_ok=True)
#    raw_state = get_peft_model_state_dict(unet_peft)
#    unet_lora_state_dict = convert_state_dict_to_diffusers(raw_state)
#    StableDiffusionPipeline.save_lora_weights(
#        save_directory=str(save_dir),
#        unet_lora_layers=unet_lora_state_dict,
#        safe_serialization=True,
#    )
#    print(f"saved LoRA to {save_dir}")


def save_lora(unet_peft, save_dir: Path):
    """diffusers が読める形式で LoRA 重みだけ保存"""
    save_dir.mkdir(parents=True, exist_ok=True)
    raw_state = peft_state_dict(unet_peft)

    # キーを diffusers 形式に整える
    converted = {}
    for k, v in raw_state.items():
        new_k = k

        # 先頭の余計なプレフィックスを全部剥がす
        while True:
            stripped = False
            for prefix in ("unet.", "base_model.model."):
                if new_k.startswith(prefix):
                    new_k = new_k[len(prefix):]
                    stripped = True
            if not stripped:
                break

        # PEFT の default アダプタ名を削除
        new_k = new_k.replace(".lora_A.default.weight", ".lora_A.weight")
        new_k = new_k.replace(".lora_B.default.weight", ".lora_B.weight")

        converted[new_k] = v

    # save_lora_weights が内部で「unet.」プレフィックスを付けてくれる
    StableDiffusionPipeline.save_lora_weights(
        save_directory=str(save_dir),
        unet_lora_layers=converted,
        safe_serialization=True,
    )
    print(f"saved LoRA to {save_dir}")





if __name__ == "__main__":
    main()
