from modal import App, Image as ModalImage, Mount, Volume, Secret, gpu, build, enter, method, asgi_app

model_name = "black-forest-labs/FLUX.1-dev"
max_train_steps = 600

instance_token = "ohwx"
instance_data_dir = "timl_images"
instance_prompt = f"photo of {instance_token} man"

# The output directory becomes the name of the final model on HuggingFace
# We include many of the parameters in the name so it's easier to identify on later
output_dir = f"/root/{instance_data_dir}_{model_name.split('/')[-1]}_dreambooth_lora_{max_train_steps}_steps_{instance_token}"

# These are optional. Set to None if not used.
class_data_dir = "flux_class_images_man"
class_prompt = "photo of man"


validation_prompt = f"photo of {instance_token} man wearing a hat"

requirements_file = "requirements_flux.txt"
training_script = "train_dreambooth_lora_flux.py"
train_dreambooth_image = (
    ModalImage.from_registry(
        "nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04", add_python="3.10"
    )
    .apt_install(["libgl1", "libglib2.0-0", "ffmpeg", "libsm6", "libxext6", "git", "nvidia-cuda-toolkit"])
    .pip_install(["prodigyopt", "wandb", "bitsandbytes", "huggingface-hub"]) #deepspeed"
    .run_commands(
        [
            "pip install -q git+https://github.com/huggingface/peft.git",
            "pip install --upgrade transformers",
            
            "git clone https://github.com/huggingface/diffusers.git /root/diffusers",
            # This PR should fix the bf16 issue: https://github.com/huggingface/diffusers/pull/9549/files
            "cd /root/diffusers && git fetch origin pull/9549/head:jeongiin && git checkout bbc91791bf41b93af64610886f792c403aac00ed",
            f"cd /root/diffusers && pip install -e .",
            f"cd /root/diffusers/examples/dreambooth && pip install -r {requirements_file}",
        ]
    )
)

app = App("dreambooth-train_2", image=train_dreambooth_image)
@app.function(
    gpu=gpu.H100(count=1), 
    timeout=86400,
    image=train_dreambooth_image,
    secrets=[Secret.from_name("huggingface-secret"), Secret.from_name("wandb-secret")],
    mounts=[Mount.from_local_dir(f"{instance_data_dir}", remote_path=f"/root/{instance_data_dir}")]
)
def train_dreambooth():
    import os
    import wandb
    from huggingface_hub import login as hf_login

    hf_login(token=os.environ["HF_TOKEN"])
    training_command = f'accelerate launch /root/diffusers/examples/dreambooth/{training_script} \
            --pretrained_model_name_or_path={model_name}  \
            --instance_data_dir={instance_data_dir} \
            --output_dir={output_dir} \
            --instance_prompt="{instance_prompt}" \
            --resolution=1024 \
            --train_batch_size=1 \
            --guidance_scale=1 \
            --gradient_accumulation_steps=4 \
            --optimizer="prodigy" \
            --learning_rate=1. \
            --lr_scheduler="constant" \
            --lr_warmup_steps=0 \
            --mixed_precision="bf16" \
            --max_train_steps={max_train_steps} \
            --seed="0" \
            --push_to_hub'


    if "WANDB_TOKEN" in os.environ:
        wandb.login(key=os.environ["WANDB_TOKEN"])
        training_command += " --report_to=wandb"
        if validation_prompt:
            training_command += f" --validation_prompt=\"{validation_prompt}\""
            training_command += " --validation_epochs=25"
    else:
        print("WANDB_TOKEN not set, not reporting to wandb")

    if class_prompt:
        training_command += f" --class_prompt=\"{class_prompt}\""
    else:
        print("Class prompt not set, skipping class prompt")
    
    os.system(training_command)