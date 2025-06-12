import mlflow
from diffusers import DiffusionPipeline
from mlflow.tracking import MlflowClient

# Step 1: Set experiment
mlflow.set_experiment("MedDream")

# Step 2: Load your trained model from local path
model_path = "/home/jovyan/datafabric/ROCOv2_Anatomical_Prompts/outputs/roco2_lora"
pipe = DiffusionPipeline.from_pretrained(model_path)

# Step 3: Start MLflow run and log artifacts
with mlflow.start_run(run_name="MedImage_SD_LoRA") as run:
    mlflow.log_param("base_model", "runwayml/stable-diffusion-v1-5")
    mlflow.log_param("steps", 3000)
    mlflow.log_artifacts(model_path, artifact_path="model")

    # Step 4: Register the model in MLflow Model Registry
    run_id = run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    model_name = "roco2_lora_model"

    client = MlflowClient()

    # Check if model already registered (to avoid duplicate error)
    try:
        client.get_registered_model(model_name)
    except:
        client.create_registered_model(model_name)

    client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=run_id
    )

print("✅ Model logged and registered to MLflow Model Registry.")
