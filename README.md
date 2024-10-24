# Checkbin ✅🗑️ - Avatar Demo
This demo uses Checkbin to evaluate different models for an AI Avatar app. AI Avatar apps use techniques like [Dreambooth](https://dreambooth.github.io/) and [LoRA](https://github.com/microsoft/LoRA). These techniques teach image models to generate a specific person, animal or object!

This demo allows you to train avatar models on top of StableDiffusion, StableDiffusion XL, and FLUX. After training, you can visualize and compare them with Checkbin's Grid ✅🗑️

![starter-screenshot](https://syntheticco.blob.core.windows.net/dreambooth-demo/avatar-demo-screenshot-framed.png)

## Step 1 - Tools
To run this demo code, you'll need auth tokens (and accounts) from the following services:
- **[Modal](www.modal.com)** - We use Modal to run the training and inference script on cloud GPUs. You can get a Modal token by signing up [here](https://modal.com/signup).
- **[HuggingFace](www.huggingface.com)** - We use HuggingFace to download models for fine-tuning. We also upload the fine-tuned models to HuggingFace. You can get a HuggingFace token by signing up [here](https://huggingface.co/join).
- **[Checkbin](www.checkbin.dev)** - We use Checkbin to compare the results of different models. You can get a Checkbin token by signing up [here](www.checkbin.dev/signup).
- **[WandB](www.wandb.ai)** (optional) - You can use WandB to track training loss. With WandB, you can also view validation images at various points in the training process. You can get a wandb token by signing up [here](https://wandb.auth0.com/login?state=hKFo2SBuZ25WcDc4YWFoZU1oNk9ZSHRXbFB6ZG54NThEeFdobqFupWxvZ2luo3RpZNkgVy1qM1d6T1ZnMTU3TmItVFA0OE5kdlgtUzVDTWRVekejY2lk2SBWU001N1VDd1Q5d2JHU3hLdEVER1FISUtBQkhwcHpJdw&client=VSM57UCwT9wbGSxKtEDGQHIKABHppzIw&protocol=oauth2&nonce=MDNvejFNNjljOUViRllhQQ%3D%3D&redirect_uri=https%3A%2F%2Fapi.wandb.ai%2Foidc%2Fcallback&response_mode=form_post&response_type=id_token&scope=openid%20profile%20email&signup=true).
- ***[AWS](aws.amazon.com), [Azure](portal.azure.com), or [GCP](cloud.google.com)** (optional) - To create class images, you'll need a cloud storage provider. For AWS, you'll need your access key and secret key. For Azure, you'll need a connection string. For GCP you'll need your JSON credential file!

Once you have tokens for each service, you'll need to add them as environment variables. Since we're running these scripts on Modal, you should add the tokens as [Modal Secrets](https://modal.com/secrets).

## Step 2 - Training
This repository contains **three training scripts** for training Dreambooth. Each works for training on top of a different type of underlying model:

### `train_dreambooth_base_sd_lora.py` 👉 StableDiffusion Base Models
This script trains Dreambooth on top of a base StableDiffusion model. Use this for any StableDiffusion 1.x or StableDiffusion 2.x models. You can also use this for models fine-tuned on top of base StableDiffusion weights, like [Realistic Vision](https://huggingface.co/SG161222/Realistic_Vision_V6.0_B1_noVAE).

### `train_dreambooth_sdxl_lora.py` 👉 Stable Diffusion XL Models
This script trains a LoRA on top of any StableDiffusionXL model. Use this for any StableDiffusionXL model, or a fine-tuned model trained on top of the SDXL weights, like [Juggernaut XL](https://huggingface.co/RunDiffusion/Juggernaut-XI-v11).

### `train_dreambooth_flux_lora.py` 👉 FLUX Models
This script trains a LoRA on top of a FLUX model. Use this with any official FLUX release, or a fine-tuned model that is trained on top of FLUX weights.

To compare different models, you should select a few different models to train on top of. Civitai's [model leaderboard](https://civitai.com/models)  and HuggingFace's [text-to-image model list](https://huggingface.co/models?pipeline_tag=text-to-image&sort=trending) are excellent resources! Once you've chosen your models, you can customize the parameters below!

### Customization Parameters
After you've selected base models, you'll need to change a few variables in the relevant script! These constants are at the top of each file:

- **model_name** - (ex. 'stable-diffusion-v1-5/stable-diffusion-v1-5) - this is the Model to train the LoRA on top of. This should be the path of the model on HuggingFace.

- **max_training_steps** - (ex. 600) - the number of training steps. Most tutorials seem to recommend 300-500. I've seen up to 2500 in practice. You can use a validation prompt to determine the necessary number of steps, as described below.

- **instance_data_dir** - (ex: '~/tim-images') - the path to the directory containing your training images. Choose 5-10 images of the subject with different angles and lighting conditions. Create a directory in the project root directory. The script will transfer this folder to Modal's GPU machine for training.

- **instance_prompt** - (ex. 'photo of sks man') - the prompt that you'll use for inference. If you're training for a man, you should choose a unique token like "sks" and use the instance prompt "photo of sks man". If your object is a dog, you should use "photo of sks dog", and so on.

- **output_dir** - (ex. 'dreambooth_on_realisic_vision_sks_1000_steps') - this is where the training process will save the trained weights. This also becomes the name of your model when it's uploaded to HuggingFace after training. If your username is 'username' and output_dir is 'dreambooth', the trained model will be uploaded to 'username/dreambooth'. I recommend including your instance key (i.e. 'sks'), the name of the base model you're training on top of (i.e. 'flux_1.0'), and other parameters you may be tuning (i.e. '1000 steps'). By default, I'm including some of these variables in the output_dir name.

- **validation_prompt** (optional - ex "photo of sks man wearing a hat") - If you're using Wandb to track training a validation prompt can be useful. This will generate a few images with that prompt after every N steps in your training cycle. If you're training on images of a man, and your instance prompt is "photo of sks man", then use a validation prompt like "photo of sks man with a hat". Validation images will help you identify the optimal number of training steps. In early training steps, the validation images won't resemble the target subject. This indicates that the model hasn't been trained enough. In later training steps, the validation images become similar to the training images. This indicates overfitting!

- **class_data_dir** (optional - ex. "~/man-class-images") - In some cases, you can improve results by including class images. If you are training on images of a dog, these would be generic images of dogs. This can help your model learn the difference between the dog you're training for and dogs in general. If you are interested in using class data, check out the Create Class Images section below!

- **class_prompt** (optional - ex. "photo of man") - If you're using class data, you should also include a class prompt. This will be the same as the instance prompt, but with the unique token removed. For example, if your instance prompt is "photo of sks man", your class prompt should be "photo of man".

Once you've set your parameters you can run the training with Modal:

```
modal run train_dreambooth_base_sd.py
```

Depending on the number of training steps, the script will run for 2-30 minutes. If you've set up WandB, the script will print a WandB URL that you can use to follow training loss! When the script completes, your model will be uploading to HuggingFace with the URL: `your_hf_username/your_output_dir`. Congrats, you've just trained your first Dreambooth LoRA!

# Step 3 - Inference + Comparison 🔬 

Once you've trained a few models, you can compare them with **Checkbin** ✅🗑️.

You'll need a token from the Checkbin Dash. If you don't already have one, create a free Checkbin account [here](www.checkbin.dev). You'll get your token after signing up. Add your token as a [Modal Secrets](https://modal.com/secrets) with the name "CHECKBIN_TOKEN".

You'll also need to create a Checkbin app, which is available from the main [Checkbin Dash](https://app.checkbin.dev/dashboard/apps). I named mine "avatar-demo".

### Part 1 - Define your test set!
To compare the Dreambooth models, you'll need a list of prompts to run inference with. I used ChatGPT to generate 20 example prompts in different styles. I attached my test prompts in 'test_prompts.json'. You can use this, or replace it with your own test cases!
```
[
    {
        "prompt": "photo of TOKEN man as a cosmic explorer standing on an alien planet, surrounded by glowing flora and strange rock formations. The style is reminiscent of Moebius and H. R. Giger, with a blend of surreal and science fiction elements, soft shading, muted colors, highly detailed landscape, atmospheric lighting, artstation, intricate textures, ethereal, full body portrait.",
        "negative_prompt": "cartoon, 3d render, low quality, blurry, pixelated, grainy, photorealistic, oversaturated, low detail, dull colors, flat lighting, amateur sketch, minimalist, blocky, childish"
    },
    ...
]
```
### Part 2 - Compare Models with `compare_dreambooth_models.py`

This script will run through the Inputs and generate an image for each input, with each model! The results are sent to Checkbin using the Checkbin SDK. After this script, you'll get a comparison grid that you can use to analyze your results. There are a few parameters that you'll need to customize to get this running for your use case!

- **checkbin_app_key** (ex. "avatar_demo") - the name of your Checkbin app. Replace this if you named your app anything other than "avatar-demo"!
- **test_prompts_path** (ex. "test_prompts.json") - the path to the JSON file containing your test prompts. Replace this if you've made your own prompts!
- **configs** - this array defines the models that you are testing! It has 5 sub-variables within it:
  - **description** (ex. "Flux1 Dev LoRA") - the name of the model you're testing. This will show up on the header of your Checkbin comparison columns!
  - **base_model_id** (ex. "black-forest-labs/FLUX.1-dev") - the path of the base model on HuggingFace. If you've trained a LoRA (the default in the scripts above), this should be the base model.
  - **pipeline_type** (ex. "flux") - the type of base model that you trained on top of. Possible values are "flux", "sdxl", and "sd". This value needs to match the training script you used above.
  - **instance_name** (ex. "sks") - the instance token. This should match the instance token you used during training.
  - **lora_model_id** (ex. "timlenardo/timl_varied_10_FLUX.1-dev_dreambooth_lora_prodigy_and_validation_600_steps_ohwx") - the path to your LoRA on HuggingFace. This should be the same as the "output_dir" in your training code.

Once you've configured these parameters and run the script, run the script with:
```
modal run compare_dreambooth_models.py
```

The program will print a Checkbin "run_id". You can load in that "run" in the [Checkbin Grid](https://app.checkbin.dev/grid) to visualize your models!

![starter-screenshot](https://syntheticco.blob.core.windows.net/dreambooth-demo/avatar-checkbin-demo-run.gif)

When the script finishes, you'll have a beautiful comparison grid on Checkbin. Hoepfully it will be very clear which models are performing best!

## Class Images (Optional)
`create_class_images.py`

Class images can help the model learn the difference between your object and other similar objects. For example, if your target object is a dog, it can be helpful to pass in photos of generic dogs.

If you'd like to try using class images, you'll need to make some class images! You can do so with the `create_class_images.py` script. You'll need to generate these images with the model that you're planning on training on top of!

These are the parameters you can set in the `create_class_images.py` script:

- **model_name** (ex. "stable-diffusion-v1-5/stable-diffusion-v1-5") - this is the model that you will use to generate your class images. This should be the same as the model that you plan to generate your class images with.
- **class_prompt** (ex. "photo of a man") - the prompt you'd like to use to generate the class images. This should be "photo of a man" if your subject is a man, "a photo of a dog" if your subject is a dog, and so on.
- **cloud_provider** (ex. "azure") - the cloud provider you want to upload the generated images to. This script supports GCP, Azure, and AWS.
- **cloud_container_name** (ex. "dreambooth-class-images") - the name of the bucket/container that you want to upload the images to.
- **number_of_images** (ex. 300) - the number of class images to generate. Other tutorials recommend 300, so I've used this as a default.

After you've run the script and generated your class images, you'll have to download them. After downloading, put them in a folder in your project's root directory. Then the scripts can transfer your images to Modal for use during training.

## Acknowledgments
This project wouldn't be possible without HuggingFace's [dreambooth training scripts](https://huggingface.co/blog/dreambooth) or Modal's infrastructure. A big thanks to the HuggingFace and Modal teams for their excellent contributions!