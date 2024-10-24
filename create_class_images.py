from modal import App, Image as ModalImage, Mount, Volume, Secret, gpu, build, enter, method, asgi_app
from upload import upload_image_to_azure, upload_image_to_gcp, upload_image_to_s3

model_name = "black-forest-labs/FLUX.1-dev"
class_prompt = "photo of man"

cloud_provider = "azure"
cloud_container_name = "dreambooth-demo"

# This is the name and key(s) of your secret in Modal
# For Azure, you will have one, the connection string
# For GCP, you will have one, the credentials JSON
# For AWS, you will have two, one for the access key and one for the secret access key
cloud_secret_name = "azure-conn-string-secret"
cloud_secret_key = "AZURE_CONNECTION_STRING"
cloud_secret_access_key = None # For AWS only

number_of_images = 300

create_class_images_image = (
    ModalImage.from_registry(
        "nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04", add_python="3.10"
    )
    .pip_install(["azure-storage-blob", "diffusers", "torch", "transformers", "accelerate", "sentencepiece"])
)

from azure.storage.blob import BlobServiceClient
import os

app = App("create-class-images", image=create_class_images_image)

@app.function(
    gpu=gpu.A100(count=1, size="80GB"),
    timeout=86400,
    image=create_class_images_image,
    secrets=[Secret.from_name(cloud_secret_name), Secret.from_name("huggingface-secret")],
    mounts=[]
)
def create_class_images():
    from diffusers import DiffusionPipeline
    import uuid
    import torch
    from huggingface_hub import login as hf_login

    hf_login(token=os.environ["HF_TOKEN"])
    pipeline = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    pipeline.to("cuda")

    for i in range(number_of_images):
        image = pipeline(class_prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
        unique_filename = f"class_image_{uuid.uuid4()}.png"
        image.save(unique_filename)

        if cloud_provider == "azure":
            upload_image_to_azure(unique_filename, os.environ[cloud_secret_key], cloud_container_name)
        elif cloud_provider == "gcp":
            # TODO: test GCP integration
            upload_image_to_gcp(unique_filename, os.environ[cloud_secret_key], cloud_container_name)
        elif cloud_provider == "s3":
            # TODO: test S3 integration
            upload_image_to_s3(unique_filename, os.environ[cloud_secret_key], os.environ[cloud_secret_access_key], cloud_container_name)
   
        os.remove(unique_filename)




