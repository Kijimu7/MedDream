from fastapi import FastAPI
from pydantic import BaseModel
from diffusers import DiffusionPipeline

# Load your model
model_path = "/home/jovyan/datafabric/ROCOv2_Anatomical_Prompts/outputs/roco2_lora"
pipe = DiffusionPipeline.from_pretrained(model_path)
pipe.to("cuda")  # or use "cpu" if no GPU

# Set up FastAPI app
app = FastAPI()

# Define input model
class PromptRequest(BaseModel):
    prompt: str

# Define endpoint
@app.post("/generate")
def generate_image(req: PromptRequest):
    image = pipe(req.prompt).images[0]
    output_path = "/home/jovyan/output.png"
    image.save(output_path)
    return {"image_path": output_path}
