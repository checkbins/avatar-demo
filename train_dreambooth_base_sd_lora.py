from modal import App, Image as ModalImage, Mount, Secret, gpu

requirements_file = "requirements.txt"
training_script = "train_dreambooth_lora.py" # or train_dreambooth.py for full Dreambooth
model_name = "stable-diffusion-v1-5/stable-diffusion-v1-5" 
max_train_steps = 2500

instance_token = "ohwx"
instance_data_dir = "timl_images"
instance_prompt = f"photo of {instance_token} man"

# The output directory becomes the name of the final model on HuggingFace
# We include many of the parameters in the name so it's easier to identify on later
output_dir = f"/root/{instance_data_dir}_{model_name.split('/')[-1]}_dreambooth_{max_train_steps}_steps_validation_{instance_token}"

# These are optional. Set to None if not used.
class_data_dir = "realistic_vision_6_class_images"
class_prompt = "photo of man"

# Validation prompt causes is a crash in this model
validation_prompt = f"photo of {instance_token} man wearing a hat"

train_dreambooth_image = (
    ModalImage.from_registry(
        "nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04", add_python="3.10"
    )
    .apt_install(["libgl1", "libglib2.0-0", "ffmpeg", "libsm6", "libxext6", "git", "nvidia-cuda-toolkit"])
    .pip_install(["prodigyopt", "wandb", "bitsandbytes", "huggingface-hub"])
    .run_commands(
        [
            # TODO versioning issues with peft and transformers. In future, may be able to install with pip instead. 
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
    gpu=gpu.A100(count=1),
    timeout=3600,
    image=train_dreambooth_image,
    secrets=[Secret.from_name("huggingface-secret"), Secret.from_name("wandb-secret")],
    mounts=[Mount.from_local_dir(f"{instance_data_dir}", remote_path=f"/root/{instance_data_dir}"),
            Mount.from_local_dir(f"{class_data_dir}", remote_path=f"/root/{class_data_dir}")]
)
def train_dreambooth():
    import os
    import wandb
    from huggingface_hub import login as hf_login

    hf_login(token=os.environ["HF_TOKEN"])
    training_command = f'python /root/diffusers/examples/dreambooth/{training_script} ' \
                       f'--pretrained_model_name_or_path={model_name} ' \
                       f'--instance_data_dir={instance_data_dir} ' \
                       f'--output_dir={output_dir} ' \
                       f'--instance_prompt="{instance_prompt}" ' \
                       f'--resolution=512 ' \
                       f'--train_batch_size=1 ' \
                       f'--train_text_encoder ' \
                       f'--gradient_accumulation_steps=1 ' \
                       f'--learning_rate=1e-4 ' \
                       f'--lr_scheduler="constant" ' \
                       f'--lr_warmup_steps=0 ' \
                       f'--max_train_steps={max_train_steps} ' \
                       f'--push_to_hub '
    
    # Optionally add validation prompt if wandb is set up 
    if "WANDB_TOKEN" in os.environ:
        wandb.login(key=os.environ["WANDB_TOKEN"])
        training_command += f'--report_to="wandb" '
        
        if validation_prompt:
            training_command += f'--validation_prompt="{validation_prompt}" '
    else: 
        print("WANDB_TOKEN not set, not reporting to wandb and skipping validation prompt")

    # Optionally add class data if it exists
    if class_data_dir and os.path.exists(f"/root/{class_data_dir}") and class_prompt:
        training_command += '--with_prior_preservation '
        training_command += f'--class_data_dir="/root/{class_data_dir}" '
        training_command += f'--class_prompt="{class_prompt}" '
    else: 
        print("Class data not set, skipping class data")

    print(training_command)
    os.system(training_command)