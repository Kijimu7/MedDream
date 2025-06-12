from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
from serve_model import generate_image

app = FastAPI()

class Prompt(BaseModel):
    prompt: str

@app.post("/generate")
def generate(req: Prompt):
    img_path = generate_image(req.prompt)
    return {"image_path": img_path}

@app.get("/image")
def get_image():
    return FileResponse("/home/jovyan/datafabric/ROCOv2_Anatomical_Prompts/result.png", media_type="image/png")
