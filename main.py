# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 2024

@author: raxephion
Perchance Revival - Recreating Old Perchance SD 1.5 Experience
Basic Stable Diffusion 1.5 Gradio App with local/Hub models and CPU/GPU selection
Added multi-image generation capability and Hires. fix.
Models from STYLE_MODEL_MAP (if Hub IDs) will now be downloaded to MODELS_DIR.

"""

import gradio as gr
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
# Import commonly used schedulers
from diffusers import DDPMScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler, LMSDiscreteScheduler
import os
from PIL import Image
import time # Optional: for timing generation
import random # Needed for random seed
import numpy as np # Needed for MAX_SEED, even if not used directly with gr.Number(-1) input
# from huggingface_hub import HfFolder # Uncomment if you need to check for HF token

# --- Configuration ---
MODELS_DIR = "checkpoints" # Directory for user's *additional* local models AND for caching Hub models
# Standard SD 1.5 sizes (multiples of 64 are generally safe) - Hires.fix removed
SUPPORTED_SD15_SIZES = ["512x512", "768x512", "512x768", "768x768", "1024x768", "768x1024", "1024x1024"]

# Mapping of friendly scheduler names to their diffusers classes
SCHEDULER_MAP = {
    "Euler": EulerDiscreteScheduler,
    "DPM++ 2M": DPMSolverMultistepScheduler,
    "DDPM": DDPMScheduler,
    "LMS": LMSDiscreteScheduler,
}
DEFAULT_SCHEDULER = "Euler"

# --- Perchance Revival Specific: Map Styles to Models ---
STYLE_MODEL_MAP = {
    "Drawn Anime": "Yntec/RevAnimatedV2Rebirth",
    "Mix Anime": "stablediffusionapi/realcartoon-anime-v11",
    "Stylized Realism V1": "Raxephion/Typhoon-SD15-V1",
    "Stylized Realism V2": "Raxephion/Typhoon-SD15-V2",
    "Realistic Humans": "stablediffusionapi/realistic-vision",
    "Semi-Realistic": "stablediffusionapi/dreamshaper8",
    "MidJourney Style": "prompthero/openjourney-v4",
    "Ghibli Style": "danyloylo/sd1.5-ghibli-style",
    "RealDream Style": "GraydientPlatformAPI/realdream11",
    "CyberRealistic" : "OmegaSunset/CyberRrealisticSD15_Diffusers"
}

DEFAULT_HUB_MODELS = [] # Keep empty as styles handle featured models

# --- Constants for UI / Generation ---
MAX_SEED = np.iinfo(np.int32).max

# --- Determine available devices and set up options ---
AVAILABLE_DEVICES = ["CPU"]
if torch.cuda.is_available():
    AVAILABLE_DEVICES.append("GPU")
    print(f"CUDA available. Found {torch.cuda.device_count()} GPU(s).")
    if torch.cuda.device_count() > 0:
        print(f"Using GPU 0: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available. Running on CPU.")

DEFAULT_DEVICE = "GPU" if "GPU" in AVAILABLE_DEVICES else "CPU"

# --- Global state for the loaded pipeline ---
current_pipeline = None
current_model_id_loaded = None
current_device_loaded = None

# --- Image Generation Function ---
def generate_image(model_input_name, selected_device_str, prompt, negative_prompt, steps, cfg_scale, scheduler_name, size, seed, num_images,
                   hires_fix_enable, denoising_strength, upscale_by): # <-- New Hires. fix parameters
    global current_pipeline, current_model_id_loaded, current_device_loaded, SCHEDULER_MAP, MAX_SEED, STYLE_MODEL_MAP, MODELS_DIR

    if not model_input_name or model_input_name == "No models found":
        raise gr.Error("No model/style selected or available. Please select a Style or add local models.")
    if not prompt:
        raise gr.Error("Please enter a prompt.")

    num_images_int = int(num_images)
    if num_images_int <= 0:
         raise gr.Error("Number of images must be at least 1.")

    device_to_use = "cuda" if selected_device_str == "GPU" and "GPU" in AVAILABLE_DEVICES else "cpu"
    if selected_device_str == "GPU" and device_to_use == "cpu":
         raise gr.Error("GPU selected but CUDA is not available to PyTorch. Ensure you have a compatible NVIDIA GPU, the correct drivers, and have installed the CUDA version of PyTorch in your environment.")

    dtype_to_use = torch.float16 if device_to_use == 'cuda' else torch.float32

    print(f"Attempting generation on device: {device_to_use}, using dtype: {dtype_to_use}")

    actual_model_id_to_load = STYLE_MODEL_MAP.get(model_input_name, model_input_name)
    print(f"Resolved model identifier: {actual_model_id_to_load}")


    if not actual_model_id_to_load or actual_model_id_to_load == "No models found":
         raise gr.Error("Invalid model selection. Could not determine which model to load.")

    if current_pipeline is None or current_model_id_loaded != actual_model_id_to_load or (current_device_loaded is not None and str(current_device_loaded) != device_to_use):
        print(f"Loading model: {actual_model_id_to_load} onto {device_to_use}...")
        # Clear memory before loading a new model
        current_pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            pipeline = StableDiffusionPipeline.from_pretrained(
                actual_model_id_to_load,
                torch_dtype=dtype_to_use,
                safety_checker=None,
                cache_dir=MODELS_DIR
            )
            pipeline = pipeline.to(device_to_use)
            current_pipeline = pipeline
            current_model_id_loaded = actual_model_id_to_load
            current_device_loaded = torch.device(device_to_use)
            print(f"Model '{actual_model_id_to_load}' loaded successfully.")

        except Exception as e:
            raise gr.Error(f"Failed to load model '{actual_model_id_to_load}': {e}. Check model name and internet connection.")


    if current_pipeline is None:
         raise gr.Error(f"Model '{actual_model_id_to_load}' failed to load previously. Cannot generate image.")

    # Set scheduler
    selected_scheduler_class = SCHEDULER_MAP.get(scheduler_name, SCHEDULER_MAP[DEFAULT_SCHEDULER])
    current_pipeline.scheduler = selected_scheduler_class.from_config(current_pipeline.scheduler.config)

    # Parse size
    try:
        w_str, h_str = size.split('x')
        width, height = int(w_str), int(h_str)
    except ValueError:
        raise gr.Error(f"Invalid size format: '{size}'. Use 'WidthxHeight' (e.g., 512x512).")

    # Set seed
    seed_int = int(seed)
    if seed_int == -1:
        seed_int = random.randint(0, MAX_SEED)
    generator = torch.Generator(device=device_to_use).manual_seed(seed_int)

    print(f"Generating {num_images_int} image(s) with seed {seed_int}...")
    start_time = time.time()

    try:
        # --- Standard text-to-image generation (pass 1) ---
        base_images = current_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=int(steps),
            guidance_scale=float(cfg_scale),
            width=width,
            height=height,
            generator=generator,
            num_images_per_prompt=num_images_int,
        ).images

        # --- HIRES FIX LOGIC (pass 2) ---
        if hires_fix_enable:
            print("Hires. fix enabled. Starting second pass...")
            
            # Use the components from the already loaded text-to-image pipeline
            pipe_img2img = StableDiffusionImg2ImgPipeline(
                vae=current_pipeline.vae,
                text_encoder=current_pipeline.text_encoder,
                tokenizer=current_pipeline.tokenizer,
                unet=current_pipeline.unet,
                scheduler=current_pipeline.scheduler,
                safety_checker=None,
                feature_extractor=current_pipeline.feature_extractor
            ).to(device_to_use)

            hires_images = []
            for i, base_image in enumerate(base_images):
                new_width = int(base_image.width * upscale_by)
                new_height = int(base_image.height * upscale_by)
                upscaled_image = base_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Re-seed the generator for each image to ensure consistency
                generator_hires = torch.Generator(device=device_to_use).manual_seed(seed_int + i)

                output_hires = pipe_img2img(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=upscaled_image,
                    strength=denoising_strength,
                    guidance_scale=float(cfg_scale),
                    generator=generator_hires,
                    num_inference_steps=int(steps),
                ).images
                hires_images.extend(output_hires)
            
            generated_images_list = hires_images
        else:
            generated_images_list = base_images

        end_time = time.time()
        print(f"Generation finished in {end_time - start_time:.2f} seconds.")
        
        return generated_images_list, seed_int

    except Exception as e:
        # Provide more specific feedback for common errors
        if "out of memory" in str(e).lower():
            raise gr.Error("Out of Memory (OOM) error. Try a smaller image size, fewer images, or disable Hires. fix.")
        raise gr.Error(f"An error occurred during generation: {e}")

# --- MODIFIED: Define the Cyberpunk CSS Theme with Orbitron font ---
cyberpunk_css = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400..900&display=swap');

/* --- CYBERPUNK HUD STYLES --- */
:root {
    --neon-blue: #00ffff;
    --neon-glow: 0 0 8px #00ffff, 0 0 16px #00ffff;
    --bg-black: #000000;
}

/* GLOBAL BACKGROUND */
body, .gradio-container {
    background: var(--bg-black) !important;
    color: var(--neon-blue) !important;
    font-family: 'Orbitron', sans-serif !important;
}

/* MAIN TITLE */
#main_title {
    color: var(--neon-blue) !important;
    text-shadow: var(--neon-glow);
    text-align: center;
    font-size: 3em !important;
    border-bottom: 1px solid var(--neon-blue);
    padding-bottom: 8px;
}

/* TEXTBOXES AND DROPDOWNS */
textarea, input[type="text"], input[type="number"], .gr-textbox, .gr-input, .gradio-dropdown {
    background: var(--bg-black) !important;
    border: 1px solid var(--neon-blue) !important;
    color: var(--neon-blue) !important;
    text-shadow: var(--neon-glow);
    font-family: 'Orbitron', sans-serif !important;
    border-radius: 0 !important;
    box-shadow: inset 0 0 12px #003333;
}

/* GALLERY BORDER */
.gradio-gallery {
    border: 1px solid var(--neon-blue) !important;
    border-radius: 0 !important;
    background: #001111 !important;
}

/* BUTTONS */
button, .gr-button {
    background: var(--bg-black) !important;
    border: 1px solid var(--neon-blue) !important;
    color: var(--neon-blue) !important;
    text-shadow: var(--neon-glow);
    border-radius: 0 !important;
    transition: all 0.2s ease-in-out;
}
button:hover, .gr-button:hover {
    background: var(--neon-blue) !important;
    color: var(--bg-black) !important;
    text-shadow: none !important;
    box-shadow: 0 0 20px #00ffff;
}

/* ACCORDIONS & BOXES */
.gr-accordion, .gr-box {
    background: var(--bg-black) !important;
    border: 1px solid var(--neon-blue) !important;
    color: var(--neon-blue) !important;
    border-radius: 0 !important;
}

/* SLIDERS */
input[type=range] {
    -webkit-appearance: none;
    background: transparent;
}
input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none;
    height: 16px;
    width: 16px;
    border-radius: 0;
    background: var(--neon-blue);
    cursor: pointer;
    margin-top: -7px;
    box-shadow: var(--neon-glow);
}
input[type=range]::-webkit-slider-runnable-track {
    width: 100%;
    height: 2px;
    cursor: pointer;
    background: var(--neon-blue) !important;
    border: 1px solid var(--neon-blue);
}

/* LABELS AND HEADERS */
h3, label, .gr-checkbox label span {
    color: var(--neon-blue) !important;
    text-shadow: var(--neon-glow);
    font-weight: 600;
    text-transform: uppercase;
}
"""

# --- Gradio Interface ---
styled_models = list(STYLE_MODEL_MAP.keys())
model_choices = styled_models if styled_models else ["No models found"]
initial_default_model = model_choices[0]

scheduler_choices = list(SCHEDULER_MAP.keys())

with gr.Blocks(css=cyberpunk_css) as demo:
    gr.Markdown("# Perchance Revival", elem_id="main_title")

    with gr.Row():
        with gr.Column(scale=2):
            model_dropdown = gr.Dropdown(
                choices=model_choices, value=initial_default_model, label="Select Style",
                interactive=bool(styled_models)
            )
            device_dropdown = gr.Dropdown(
                choices=AVAILABLE_DEVICES, value=DEFAULT_DEVICE, label="Processing Device",
                interactive=len(AVAILABLE_DEVICES) > 1
            )
            prompt_input = gr.Textbox(label="Positive Prompt", placeholder="Enter your prompt...", lines=3)
            negative_prompt_input = gr.Textbox(label="Negative Prompt", placeholder="e.g. blurry, bad quality...", lines=2)

            generate_button = gr.Button("✨ Generate Image ✨", variant="primary")

            with gr.Accordion("Advanced Settings", open=False):
                steps_slider = gr.Slider(minimum=5, maximum=150, value=30, label="Inference Steps", step=1)
                cfg_slider = gr.Slider(minimum=1.0, maximum=30.0, value=7.5, label="CFG Scale", step=0.1)
                scheduler_dropdown = gr.Dropdown(choices=scheduler_choices, value=DEFAULT_SCHEDULER, label="Scheduler")
                size_dropdown = gr.Dropdown(choices=SUPPORTED_SD15_SIZES, value="512x768", label="Image Size")
                seed_input = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                num_images_slider = gr.Slider(minimum=1, maximum=9, value=1, step=1, label="Number of Images")

            # --- New Hires. fix Section ---
            with gr.Accordion("Hires. fix", open=False):
                hires_fix_enable_checkbox = gr.Checkbox(label="Enable Hires. fix", value=False)
                denoising_strength_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.01, label="Denoising Strength")
                upscale_by_slider = gr.Slider(minimum=1.0, maximum=2.0, value=1.5, step=0.05, label="Upscale by")

        with gr.Column(scale=3):
            # --- MODIFIED LINE IS HERE ---
            output_gallery = gr.Gallery(label="Generated Images", show_label=True, interactive=False, format="png", columns=4, object_fit="contain")
            actual_seed_output = gr.Number(label="Actual Seed Used", precision=0, interactive=False)

    generate_button.click(
        fn=generate_image,
        inputs=[
            model_dropdown, device_dropdown, prompt_input, negative_prompt_input,
            steps_slider, cfg_slider, scheduler_dropdown, size_dropdown,
            seed_input, num_images_slider,
            # Pass new Hires. fix inputs
            hires_fix_enable_checkbox, denoising_strength_slider, upscale_by_slider
        ],
        outputs=[output_gallery, actual_seed_output],
        api_name="generate"
    )
    
    gr.Markdown(f"Models are downloaded to the `{MODELS_DIR}` folder.")

if __name__ == "__main__":
    print("--- Starting Perchance Revival ---")
    print(f"Models will be cached in: {os.path.abspath(MODELS_DIR)}")
    demo.launch(show_error=True, inbrowser=True)
    print("Gradio interface closed.")
