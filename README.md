# 🧠 MedDream: AI-Powered Visualizations for Medical Text

**MedDream** transforms structured medical text into high-quality medical images using fine-tuned diffusion models. Created for the [HP & NVIDIA AI Studio Developer Challenge](https://hpaistudio.devpost.com/), this project enables safe, privacy-preserving medical image generation for education, training, and research.

---

## 🚀 Project Summary

MedDream takes a CSV of medical image captions and their associated image files and uses LoRA fine-tuning with Hugging Face `diffusers` to create a domain-specific Stable Diffusion model.

---

## 📁 Project Structure

meddream-roco-diffusers/
├── datafabric/
│ └── ROCOv2_Anatomical_Prompts/
│ ├── train_images/
│ └── train_data_for_diffusers.jsonl
├── outputs/
│ └── roco2_lora/
├── diffusers/
│ └── examples/
│ └── text_to_image/
│ └── train_text_to_image.py
├── scripts/
│ └── make_jsonl.py
├── README.md
└── requirements.txt


---

## 🛠️ Setup Instructions

### 1. Clone the Repo and Set Up Environment

```bash
git clone https://github.com/YOUR_USERNAME/meddream-roco-diffusers.git
cd meddream-roco-diffusers

python -m venv venv-diffusers
source venv-diffusers/bin/activate

pip install -r requirements.txt


### Prepare the Dataset
datafabric/ROCOv2_Anatomical_Prompts/
├── train_images/
│   └── ROCOv2_2023_train_000001.jpg
├── train_captions.csv

python scripts/make_jsonl.py


python diffusers/examples/text_to_image/train_text_to_image.py \
  --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
  --train_data_dir "/home/jovyan/datafabric/ROCOv2_Anatomical_Prompts" \
  --resolution 512 \
  --train_batch_size 4 \
  --max_train_steps 1000 \
  --learning_rate 5e-6 \
  --mixed_precision fp16 \
  --output_dir "/home/jovyan/outputs/roco2_lora"


from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained("/home/jovyan/outputs/roco2_lora", torch_dtype=torch.float16)
pipe.to("cuda")

prompt = "Axial brain CT showing left-sided epidural hematoma"
image = pipe(prompt).images[0]
image.save("output_sample.png")




