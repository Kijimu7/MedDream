from diffusers import DiffusionPipeline

model_path = "/home/jovyan/datafabric/ROCOv2_Anatomical_Prompts/outputs/roco2_lora"
pipe = DiffusionPipeline.from_pretrained(model_path)
pipe.to("cuda")  # Change to "cpu" if you're not using GPU
prompt = "CT showing signs of stroke"
image = pipe(prompt).images[0]
image.save("result.png")
print("Saved image as result.png")

