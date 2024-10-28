class_image_dir = "realistic_vision_V5_class_images"

# I've created man and woman class images for many models. You can download them with this script!
class_image_hf_repo = "timlenardo/realistic_Vision_V5.1_noVAE_woman_class_images"

# Realistic Vision V5.1 noVAE
# timlenardo/realistic_Vision_V5.1_noVAE_woman_class_images
# timlenardo/realistic_Vision_V5.1_noVAE_man_class_images

# Playground V2.5 1024px Aesthetic
# timlenardo/playground-v2.5-1024px-aesthetic_woman_class_images
# timlenardo/playground-v2.5-1024px-aesthetic_man_class_images

# DreamShaper
# timlenardo/DreamShaper_woman_class_images
# timlenardo/DreamShaper_man_class_images

# Absolute Reality v1.8.1
# timlenardo/AbsoluteReality_v1.8.1_woman_class_images
# timlenardo/AbsoluteReality_v1.8.1_man_class_images

# JuggernautXL
# timlenardo/Juggernaut-XL-v6_woman_class_images
# timlenardo/Juggernaut-XL-v6_man_class_images

# RealVisXL V4.0
# timlenardo/RealVisXL_V4.0_woman_class_images
# timlenardo/RealVisXL_V4.0_man_class_images

# Stable Diffusion XL Base 1.0
# timlenardo/stable-diffusion-xl-base-1.0_woman_class_images
# timlenardo/stable-diffusion-xl-base-1.0_man_class_images

# Stable Diffusion 3.5 Large
# timlenardo/stable-diffusion-3.5-large_woman_class_images
# timlenardo/stable-diffusion-3.5-large_man_class_images

# Flux 1.0 Dev
# timlenardo/FLUX.1-dev_woman_class_images
# timlenardo/FLUX.1-dev_man_class_images


import os

# Create the directory if it doesn't exist
os.makedirs(class_image_dir, exist_ok=True)

from datasets import load_dataset

# Load the dataset from the Hugging Face repository

dataset = load_dataset(path=class_image_hf_repo)

# Iterate over the dataset and save images to the specified directory
for i, example in enumerate(dataset['train']):
    image = example['image']
    image_path = os.path.join(class_image_dir, f"class_image_{i}.png")
    image.save(image_path)
    print(f"Saved image {i} to {image_path}")

