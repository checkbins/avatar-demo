from modal import App, Image as ModalImage, Mount, Volume, Secret, gpu, build, enter, method, asgi_app
import os, sys, uuid
from upload import upload_image_to_azure, upload_image_to_gcp, upload_image_to_s3
from collections import namedtuple

checkbin_app_key = "avatar_demo"
test_prompts_path = "test_prompts.json"

DreamboothInferenceConfig = namedtuple(
    'DreamboothInferenceConfig', 
    [
        'description', # This is your name for the model, it can be anything
        'base_model_id', # This is the model id, it should be the path of the model you trained on top of
        'pipeline_type', # This is the type of model you trained on top of
        'instance_name', # This is the instance name you used for training
        'lora_model_id' # This is the model you trained!
    ]
)

# This is an example config for models we've trained. You should change this to your own models.!
configs = [
    DreamboothInferenceConfig(
        "Flux1 Dev LoRA", 
        "black-forest-labs/FLUX.1-dev", 
        "flux", 
        "ohwx", 
        "timlenardo/timl_varied_10_FLUX.1-dev_dreambooth_lora_prodigy_and_validation_600_steps_ohwx" 
    ),
    DreamboothInferenceConfig(
        "Pony Diffusion V6 XL Turbo DPO LoRA", 
        "Bakanayatsu/ponyDiffusion-V6-XL-Turbo-DPO", 
        "sdxl", 
        "ohwx", 
        'timlenardo/timl_10_ponydiffusion_v6_xl_turbo_dpo_dreambooth_class_prompt_fix'
    ),
    DreamboothInferenceConfig(
        "Realistic Vision Model 6.0 w Validation, 1000 steps", 
        "timlenardo/timl_varied_10_realistic_vision_v6.0_B1_noVAE_dreambooth_1500_steps_validation_ohwx", 
        "sd", 
        "ohwx",
        None
    )
]

test_dreambooth_image = (
    ModalImage.from_registry(
        "nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04", add_python="3.10"
    )
    .apt_install(
        [
            "git", 
            "libgl1",
            "libglib2.0-0",
            "libsm6",
            "libxext6", 
            "ffmpeg",
        ]
    )
    .pip_install(
        [
            "transformers",
            "diffusers",
            "torch",
            "numpy",
            "huggingface_hub",
            "accelerate",
            "sentencepiece",
            "peft",
            "opencv-python",
            "boto3",
            "google-cloud-storage",
            "azure-storage-blob",
            "azure-storage-file-datalake",
            "pydantic",
            "tinydb",
        ]
    )
    # TODO everything after "opencv-python" are needed for the local version of checkbin, but can be removed when we move to the pip packages
)

app = App("dreambooth-demo", image=test_dreambooth_image)

def run_dreambooth_inference(checkbins, checkin_name, model_id, model_type, token, lora_id=None):
    import torch
    from diffusers import FluxPipeline, StableDiffusionPipeline, StableDiffusionXLPipeline
    
    if model_type == "flux":
        print(f"Loading Flux model from {model_id}")
        pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    elif model_type == "sd":
        print(f"Loading Stable Diffusion model from {model_id}")
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    elif model_type == "sdxl":
        print(f"Loading Stable Diffusion XL model from {model_id}")
        pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    if lora_id is not None:
        pipe.load_lora_weights(lora_id, weight_name="pytorch_lora_weights.safetensors", adapter_name="skstl-lora")
        pipe.set_adapters(["skstl-lora"], [1.0])
    else:
        print("No LoRA weights provided for this model.")
    
    for checkbin in checkbins:
        negative_prompt = checkbin.get_input_data('negative_prompt')
        prompt = checkbin.get_input_data('prompt')
        prompt = prompt.replace("TOKEN", token)

        checkbin.checkin(checkin_name)
        if model_type == "flux":
            image = pipe(
                prompt=prompt
            ).images[0]
        else:
            image = pipe(
                prompt=prompt, 
                negative_prompt=negative_prompt,
                num_inference_steps=50, 
                guidance_scale=7.5
            ).images[0]
        image.save("inference_output.png")
        checkbin.upload_file(
            "inference_output",
            "inference_output.png", 
            "image"
        )
        os.remove("inference_output.png")


@app.function(
    gpu=gpu.A100(size="80GB"),
    timeout=3600,
    image=test_dreambooth_image,
    secrets=[Secret.from_name("huggingface-secret"), Secret.from_name("checkbin-secret")],
    mounts=[Mount.from_local_dir("./checkbin-python", remote_path="/root/checkbin-python"),
            Mount.from_local_dir("./inputs", remote_path="/root/inputs")]
)
def test_dreambooth():
    from huggingface_hub import login
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, FluxPipeline
    
    # TODO migrate to use the python package instead of local checkbin-python
    sys.path.insert(0, 'checkbin-python/src')
    import checkbin

    login(token=os.environ["HF_TOKEN"])
    checkbin.authenticate(token=os.environ["CHECKBIN_TOKEN"])
    checkbin_app = checkbin.App(app_key=checkbin_app_key, mode="remote")
    checkbin_input_set = checkbin_app.create_input_set("20x Test Prompts")
    input_set_id = checkbin_input_set.create_from_json(json_file='./inputs/test_prompts.json')

    bins = checkbin_app.start_run(set_id=input_set_id)

    for config in configs:
        run_dreambooth_inference(bins, config.description, config.base_model_id, config.pipeline_type, config.instance_name, config.lora_model_id)

    for checkbin in bins:
        checkbin.submit()
