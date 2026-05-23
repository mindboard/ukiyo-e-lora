
# Ukiyo-e Stable Diffusion + LoRA Training

## Usage

Create a venv environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required libraries

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Run the training

```bash
sh train.sh
```

Generate ukiyo-e images using the training results

```bash
python generate_image.py examples/input-1.png output-1.png
```

