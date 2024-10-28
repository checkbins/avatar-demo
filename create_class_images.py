from modal import App, Image as ModalImage, Mount, Volume, Secret, gpu, build, enter, method, asgi_app
from upload import upload_image_to_azure, upload_image_to_gcp, upload_image_to_s3

model_names = [
    "black-forest-labs/FLUX.1-dev", 
    "stabilityai/stable-diffusion-3.5-large", 
    "stabilityai/stable-diffusion-3-medium", 
    "stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/stable-diffusion-1-5",
    "SG161222/RealVisXL_V4.0",
    "RunDiffusion/Juggernaut-XL-v6",
    "Lykon/DreamShaper",
    "digiplay/AbsoluteReality_v1.8.1",
    "playgroundai/playground-v2.5-1024px-aesthetic",
    "SG161222/Realistic_Vision_V5.1_noVAE",

]
class_prompts = ["photo of man", "photo of woman"]
dataset_names = ["_man_class_images", "_woman_class_images"]
huggingface_username = "timlenardo"
number_of_images = 300

create_class_images_image = (
    ModalImage.from_registry(
        "nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04", add_python="3.10"
    )
    .pip_install(["diffusers", "torch", "transformers", "accelerate", "sentencepiece", "datasets"])
)

# from azure.storage.blob import BlobServiceClientd
import os

app = App("create-class-images", image=create_class_images_image)

@app.function(
    gpu=gpu.A100(count=1, size="80GB"),
    timeout=86400,
    image=create_class_images_image,
    secrets=[Secret.from_name("huggingface-secret")],
    mounts=[]
)
def create_class_images():
    from diffusers import DiffusionPipeline
    import torch
    from huggingface_hub import login as hf_login
    from datasets import Dataset, load_dataset, Image as DatasetImage
    import pandas as pd
    import uuid

    hf_login(token=os.environ["HF_TOKEN"])
    for model_name in model_names:
        try:
            pipeline = DiffusionPipeline.from_pretrained(
                model_name, 
                torch_dtype=torch.float16,
                num_inference_steps=50,  # Override the number of steps
                guidance_scale=7.5       # Override the CFG (Classifier-Free Guidance)
            )
            pipeline.to("cuda")

            os.makedirs("/root/generated_images", exist_ok=True)
            for class_prompt, dataset_name in zip(class_prompts, dataset_names):
                image_paths = []
                for i in range(number_of_images):
                    image = pipeline(class_prompt, num_inference_steps=30, guidance_scale=7.5).images[0]

                    unique_filename = f"/root/generated_images/class_image_{uuid.uuid4()}.png"
                    image.save(unique_filename)
                    image_paths.append(unique_filename)

                dataset = load_dataset("imagefolder", data_dir="/root/generated_images")
                model_name_split = model_name.split("/")
                model_name_last_value = model_name_split[-1]
                dataset.push_to_hub(f"{huggingface_username}/{model_name_last_value}_{dataset_name}")

                for image_path in image_paths:
                    os.remove(image_path)
        except Exception as e:
            print(f"An error occurred: {e}")