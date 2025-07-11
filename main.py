# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 2024

@author: raxephion
Perchance Revival - Recreating Old Perchance SD 1.5 Experience
Basic Stable Diffusion 1.5 Gradio App with local/Hub models and CPU/GPU selection
Added multi-image generation capability.
Models from STYLE_MODEL_MAP (if Hub IDs) will now be downloaded to MODELS_DIR.

NOTE: App still in early development - UI will be adjusted to match Perchance presets
"""

import gradio as gr
import torch
from diffusers import StableDiffusionPipeline
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
# Standard SD 1.5 sizes (multiples of 64 are generally safe)
SUPPORTED_SD15_SIZES = ["512x512", "768x512", "512x768", "768x768", "1024x768", "768x1024", "1024x1024", "hire.fix"]

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
    "Realism" : "OmegaSunset/CyberRrealisticSD15_Diffusers"
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

# --- Helper function to list available local models ---
def list_local_models(models_dir_param): # Renamed param
    if not os.path.exists(models_dir_param):
        os.makedirs(models_dir_param)
        print(f"Created directory: {models_dir_param}")
        return []
    local_models = [os.path.join(models_dir_param, d) for d in os.listdir(models_dir_param)
                    if os.path.isdir(os.path.join(models_dir_param, d))]
    return local_models

# --- Image Generation Function ---
def generate_image(model_input_name, selected_device_str, prompt, negative_prompt, steps, cfg_scale, scheduler_name, size, seed, num_images):
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

    dtype_to_use = torch.float32
    if device_to_use == "cuda":
        if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 7:
             dtype_to_use = torch.float16
             print("GPU supports FP16, using torch.float16 for potential performance/memory savings.")
        else:
             dtype_to_use = torch.float32
             print("GPU might not fully support FP16 or capability check failed, using torch.float32.")
    else:
         dtype_to_use = torch.float32

    print(f"Attempting generation on device: {device_to_use}, using dtype: {dtype_to_use}")

    actual_model_id_to_load = None
    if model_input_name in STYLE_MODEL_MAP:
        actual_model_id_to_load = STYLE_MODEL_MAP[model_input_name]
        print(f"Selected style '{model_input_name}' maps to model: {actual_model_id_to_load}")
    elif model_input_name.startswith("Local: "):
        actual_model_id_to_load = model_input_name.replace("Local: ", "", 1)
        print(f"Selected local model path: {actual_model_id_to_load}")
    else:
        actual_model_id_to_load = model_input_name
        print(f"Selected identifier '{model_input_name}' not a style or local path format, attempting to load as raw ID/path...")

    if not actual_model_id_to_load or actual_model_id_to_load == "No models found":
         raise gr.Error("Invalid model selection. Could not determine which model to load.")

    if current_pipeline is None or current_model_id_loaded != actual_model_id_to_load or (current_device_loaded is not None and str(current_device_loaded) != device_to_use):
        print(f"Loading model: {actual_model_id_to_load} onto {device_to_use}...")
        if current_pipeline is not None:
             print(f"Unloading previous model '{current_model_id_loaded}' from {current_device_loaded}...")
             if str(current_device_loaded) == "cuda":
                  try:
                      current_pipeline.to("cpu")
                      print("Moved previous pipeline to CPU.")
                  except Exception as move_e:
                      print(f"Warning: Failed to move previous pipeline to CPU: {move_e}")
             del current_pipeline
             current_pipeline = None
             if str(current_device_loaded) == "cuda":
                 try:
                     torch.cuda.empty_cache()
                     print("Cleared CUDA cache.")
                 except Exception as cache_e:
                     print(f"Warning: Error clearing CUDA cache: {cache_e}")

        if device_to_use == "cuda":
             if not torch.cuda.is_available():
                  raise gr.Error("CUDA selected but not available. Please install PyTorch with CUDA support or select CPU.")

        try:
            is_local_path_check = os.path.isdir(actual_model_id_to_load)

            if is_local_path_check:
                 print(f"Attempting to load local model from: {actual_model_id_to_load}")
                 pipeline = StableDiffusionPipeline.from_pretrained(
                     actual_model_id_to_load,
                     torch_dtype=dtype_to_use,
                     safety_checker=None,
                 )
            else: # This is the block for Hub models (or model IDs from STYLE_MODEL_MAP that are not local paths)
                 print(f"Attempting to load Hub model: {actual_model_id_to_load} into local cache: {MODELS_DIR}")
                 pipeline = StableDiffusionPipeline.from_pretrained(
                     actual_model_id_to_load,
                     torch_dtype=dtype_to_use,
                     safety_checker=None,
                     cache_dir=MODELS_DIR  # <--- THIS IS THE ADDED/MODIFIED LINE
                 )

            pipeline = pipeline.to(device_to_use)
            current_pipeline = pipeline
            current_model_id_loaded = actual_model_id_to_load
            current_device_loaded = torch.device(device_to_use)

            unet_config = getattr(pipeline, 'unet', None)
            if unet_config and hasattr(unet_config, 'config') and hasattr(unet_config.config, 'cross_attention_dim'):
                 cross_attn_dim = unet_config.config.cross_attention_dim
                 if cross_attn_dim != 768:
                     warning_msg = (f"Warning: Loaded model '{actual_model_id_to_load}' might not be a standard SD 1.x model "
                                    f"(expected UNet cross_attention_dim 768, found {cross_attn_dim}). "
                                    "Results may be unexpected or generation might fail.")
                     print(warning_msg)
                     gr.Warning(warning_msg)
                 else:
                     print("UNet cross_attention_dim is 768, consistent with SD 1.x.")
            else:
                 print("Could not check UNet cross_attention_dim.")

            print(f"Model '{actual_model_id_to_load}' loaded successfully on {current_device_loaded} with dtype {dtype_to_use}.")

        except Exception as e:
            current_pipeline = None
            current_model_id_loaded = None
            current_device_loaded = None
            print(f"Error loading model '{actual_model_id_to_load}': {e}")
            error_message_lower = str(e).lower()
            if "require users to upgrade torch to at least v2.6" in error_message_lower or "vulnerability issue in `torch.load`" in error_message_lower:
                 print("\n--- HINT: PyTorch version likely too old for this model/library version ---")
                 print("The model uses a file format requiring a newer PyTorch.")
                 print("You need PyTorch 2.6 or higher.")
                 print("Visit https://pytorch.org/get-started/locally/ to find the exact command for your system and CUDA version.")
                 print("You will need to manually run that command while the virtual environment (venv) is active.")
                 print("------------------------------------------------------------------------\n")
                 raise gr.Error(
                     f"Failed to load model '{actual_model_id_to_load}': Your installed PyTorch version is too old "
                     f"for this model file format. You need PyTorch 2.6 or higher. "
                     f"Please manually install an updated PyTorch version. "
                     f"See instructions on the PyTorch website: https://pytorch.org/get-started/locally/. Error: {e}"
                 )
            elif "cannot find requested files" in error_message_lower or "404 client error" in error_message_lower or "no such file or directory" in error_message_lower:
                 raise gr.Error(f"Model '{actual_model_id_to_load}' not found. Check name/path, Hugging Face Hub ID spelling, or internet connection. Error: {e}")
            elif "checkpointsnotfounderror" in error_message_lower or "valueerror: could not find a valid model structure" in error_message_lower:
                 raise gr.Error(f"No valid diffusers model at '{actual_model_id_to_load}'. Ensure it's a diffusers format directory or a valid Hub ID. Error: {e}")
            elif "out of memory" in error_message_lower:
                 raise gr.Error(f"Out of Memory (OOM) loading model '{actual_model_id_to_load}'. Try a lighter model or select CPU. Error: {e}")
            elif "cusolver64" in error_message_lower or "cuda driver version" in error_message_lower or "cuda error" in error_message_lower:
                 raise gr.Error(f"CUDA/GPU Driver Error: {e} loading '{actual_model_id_to_load}'. Check drivers, PyTorch with CUDA installation, or select CPU.")
            elif "safetensors_rust.safetensorserror" in error_message_lower or "oserror: cannot load" in error_message_lower or "filenotfounderror" in error_message_lower:
                 raise gr.Error(f"Model file error for '{actual_model_id_to_load}': {e}. Files might be corrupt, incomplete, or the path is wrong.")
            elif "could not import" in error_message_lower or "module not found" in error_message_lower:
                 raise gr.Error(f"Dependency error: {e} during model loading. Ensure all dependencies are installed (run setup.bat) and PyTorch is installed correctly for your device.")
            else:
                raise gr.Error(f"Failed to load model '{actual_model_id_to_load}': An unexpected error occurred. {e}")

    if current_pipeline is None:
         raise gr.Error(f"Model '{actual_model_id_to_load}' failed to load previously. Cannot generate image.")

    selected_scheduler_class = SCHEDULER_MAP.get(scheduler_name)
    if selected_scheduler_class is None:
         print(f"Warning: Unknown scheduler '{scheduler_name}'. Using default: {DEFAULT_SCHEDULER}.")
         selected_scheduler_class = SCHEDULER_MAP[DEFAULT_SCHEDULER]
         gr.Warning(f"Unknown scheduler '{scheduler_name}'. Using default: {DEFAULT_SCHEDULER}.")

    try:
        scheduler_config = current_pipeline.scheduler.config
        current_pipeline.scheduler = selected_scheduler_class.from_config(scheduler_config)
        print(f"Scheduler set to: {scheduler_name}")
    except Exception as e:
        print(f"Error setting scheduler '{scheduler_name}': {e}")
        try:
             print(f"Attempting to fallback to default scheduler: {DEFAULT_SCHEDULER}")
             current_pipeline.scheduler = SCHEDULER_MAP[DEFAULT_SCHEDULER].from_config(scheduler_config)
             gr.Warning(f"Failed to set scheduler to '{scheduler_name}', fell back to {DEFAULT_SCHEDULER}. Error: {e}")
        except Exception as fallback_e:
             print(f"Fallback scheduler failed too: {fallback_e}")
             raise gr.Error(f"Failed to configure scheduler '{scheduler_name}' and fallback failed. Error: {fallback_e} (Original: {e})")

    width, height = 512, 512
    if size.lower() == "hire.fix":
        width, height = 1024, 1024
        print(f"Interpreting 'hire.fix' size as {width}x{height}")
    else:
        try:
            w_str, h_str = size.split('x')
            width = int(w_str)
            height = int(h_str)
        except ValueError:
            raise gr.Error(f"Invalid size format: '{size}'. Use 'WidthxHeight' (e.g., 512x512) or 'hire.fix'.")
        except Exception as e:
             raise gr.Error(f"Error parsing size '{size}': {e}")

    multiple_check = 64
    if width % multiple_check != 0 or height % multiple_check != 0:
         warning_msg_size = (f"Warning: Image size {width}x{height} is not a multiple of {multiple_check}. "
                             f"Stable Diffusion 1.5 models are typically trained on sizes like 512x512 or 768x768. "
                             "Using non-standard sizes may cause tiling, distortions, or other artifacts.")
         print(warning_msg_size)
         gr.Warning(warning_msg_size)

    generator = None
    generator_device = current_pipeline.device if current_pipeline else torch.device(device_to_use)
    seed_int = int(seed)

    if seed_int == -1:
        seed_int = random.randint(0, MAX_SEED)
        print(f"User requested random seed (-1), generated: {seed_int}")
    else:
        print(f"Using provided seed: {seed_int}")

    try:
        generator = torch.Generator(device=generator_device).manual_seed(seed_int)
        print(f"Generator set with seed {seed_int} on device: {generator_device}")
    except Exception as e:
         print(f"Warning: Error setting seed generator on device {generator_device}: {e}. Falling back to default generator or system random.")
         gr.Warning(f"Failed to set seed generator with seed {seed_int}. Using random seed. Error: {e}")
         generator = None
         pass

    num_images_int = int(num_images)
    print(f"Generating {num_images_int} image(s) for Style/Model '{model_input_name}' (Actual Model: {actual_model_id_to_load}): Prompt='{prompt[:80]}{'...' if len(prompt) > 80 else ''}', NegPrompt='{negative_prompt[:80]}{'...' if len(negative_prompt) > 80 else ''}', Steps={int(steps)}, CFG={float(cfg_scale)}, Size={width}x{height}, Scheduler={scheduler_name}, Seed={seed_int if generator else 'System Random'}, Images={num_images_int}")
    start_time = time.time()

    try:
        num_inference_steps_int = int(steps)
        guidance_scale_float = float(cfg_scale)

        if num_inference_steps_int <= 0 or guidance_scale_float <= 0:
             raise ValueError("Steps and CFG Scale must be positive values.")
        if width <= 0 or height <= 0:
             raise ValueError("Image width and height must be positive.")

        output = current_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            num_inference_steps=num_inference_steps_int,
            guidance_scale=guidance_scale_float,
            width=width,
            height=height,
            generator=generator,
            num_images_per_prompt=num_images_int,
        )
        end_time = time.time()
        print(f"Generation finished in {end_time - start_time:.2f} seconds.")
        generated_images_list = output.images
        actual_seed_used = seed_int
        return generated_images_list, actual_seed_used

    except gr.Error as e:
         raise e
    except ValueError as ve:
         print(f"Parameter Error: {ve}")
         raise gr.Error(f"Invalid Parameter: {ve}")
    except Exception as e:
        print(f"An error occurred during image generation: {e}")
        error_message_lower = str(e).lower()
        if "size must be a multiple of" in error_message_lower or "invalid dimensions" in error_message_lower or "shape mismatch" in error_message_lower:
             raise gr.Error(f"Image generation failed - Invalid size '{width}x{height}' for model: {e}. Try a multiple of 64 or 8.")
        elif "out of memory" in error_message_lower or "cuda out of memory" in error_message_lower:
             print("Hint: Try smaller image size, fewer steps, fewer images, or a model that uses less VRAM.")
             raise gr.Error(f"Out of Memory (OOM) during generation. Try smaller size/steps, fewer images, or select CPU. Error: {e}")
        elif "runtimeerror" in error_message_lower:
             raise gr.Error(f"Runtime Error during generation: {e}. This could be a model/scheduler incompatibility or other issue.")
        elif "device-side assert" in error_message_lower or "cuda error" in error_message_lower:
             raise gr.Error(f"CUDA/GPU Error during generation: {e}. Ensure PyTorch with CUDA is correctly installed and compatible.")
        elif "expected all tensors to be on the same device" in error_message_lower:
             raise gr.Error(f"Device mismatch error during generation: {e}. This is an internal error, please report it.")
        else:
             raise gr.Error(f"Image generation failed: An unexpected error occurred. {e}")

# --- Gradio Interface ---
local_models_list = list_local_models(MODELS_DIR) # Use the renamed param for the function call
styled_models = list(STYLE_MODEL_MAP.keys())
additional_local_model_names = [f"Local: {path}" for path in local_models_list] # Use the renamed list

model_choices = styled_models + additional_local_model_names

if not model_choices:
    initial_model_choices = ["No models found"]
    initial_default_model = "No models found"
    model_dropdown_interactive = False
    print(f"\n--- IMPORTANT ---")
    print(f"No models available!")
    print(f"Please define Styles and their corresponding Hub models in STYLE_MODEL_MAP in main.py")
    print(f"or place additional local diffusers models in '{os.path.abspath(MODELS_DIR)}'.") # Uses MODELS_DIR
    print(f"-----------------\n")
else:
    initial_model_choices = model_choices
    if styled_models:
         initial_default_model = styled_models[0]
    elif additional_local_model_names:
         initial_default_model = additional_local_model_names[0]
    else:
         initial_default_model = model_choices[0]
    model_dropdown_interactive = True

scheduler_choices = list(SCHEDULER_MAP.keys())

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        f"""
        # Perchance Revival
        Recreating the experience of the old Perchance Stable Diffusion generator.
        Select a style below, enter your prompt, and generate!
        Models selected by Style will be downloaded to the `{MODELS_DIR}` folder if not already present.
        Note: this app is currently in early development - UI will be adjusted to match Perchance presets.
        Note: 'hire.fix' size option currently generates at 1024x1024.
        Have fun!
        """ # Added note about MODELS_DIR
    )

    with gr.Row():
        with gr.Column(scale=2):
            model_dropdown = gr.Dropdown(
                choices=initial_model_choices,
                value=initial_default_model,
                label=f"Select Style / Model (Featured Styles or Additional Local from ./{MODELS_DIR})", # Uses MODELS_DIR
                interactive=model_dropdown_interactive,
            )
            device_dropdown = gr.Dropdown(
                choices=AVAILABLE_DEVICES,
                value=DEFAULT_DEVICE,
                label="Processing Device",
                interactive=len(AVAILABLE_DEVICES) > 1,
            )
            prompt_input = gr.Textbox(label="Positive Prompt", placeholder="Enter your prompt...", lines=3, autofocus=True)
            negative_prompt_input = gr.Textbox(label="Negative Prompt (Optional)", placeholder="Enter negative prompt (e.g. blurry, bad quality)...", lines=2)

            with gr.Accordion("Advanced Settings", open=False):
                with gr.Row():
                    steps_slider = gr.Slider(minimum=5, maximum=150, value=30, label="Inference Steps", step=1)
                    cfg_slider = gr.Slider(minimum=1.0, maximum=30.0, value=7.5, label="CFG Scale", step=0.1)
                with gr.Row():
                     scheduler_dropdown = gr.Dropdown(
                        choices=scheduler_choices,
                        value=DEFAULT_SCHEDULER,
                        label="Scheduler"
                    )
                     size_dropdown = gr.Dropdown(
                        choices=SUPPORTED_SD15_SIZES,
                        value="512x768",
                        label="Image Size"
                    )
                with gr.Row():
                     seed_input = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                     num_images_slider = gr.Slider(
                         minimum=1,
                         maximum=4,
                         value=1,
                         step=1,
                         label="Number of Images",
                         interactive=True
                     )

            generate_button = gr.Button("✨ Generate Image ✨", variant="primary", scale=1)

        with gr.Column(scale=3):
            output_gallery = gr.Gallery(
                label="Generated Images",
                show_label=True,
                show_share_button=True,
                show_download_button=True,
                interactive=False
            )
            actual_seed_output = gr.Number(label="Actual Seed Used", precision=0, interactive=False)

    generate_button.click(
        fn=generate_image,
        inputs=[
            model_dropdown,
            device_dropdown,
            prompt_input,
            negative_prompt_input,
            steps_slider,
            cfg_slider,
            scheduler_dropdown,
            size_dropdown,
            seed_input,
            num_images_slider
        ],
        outputs=[output_gallery, actual_seed_output],
        api_name="generate"
    )

    gr.Markdown(
        f"""
        ---
        **Usage Notes:**

        1. Select a Style from the dropdown.

        2. Enter your positive and optional negative prompts.
        3. Optional: Adjust advanced settings (Steps, CFG Scale, Scheduler, Size, Seed, Number of Images).
        4. Click "Generate Image".
        5. Have fun!
        6. The first generation with a new Style/Model might take some time to load as the model is initially downloaded from the hub to your local `{MODELS_DIR}` folder.
        """ # Uses MODELS_DIR
    )

if __name__ == "__main__":
    print("\n--- Starting Perchance Revival ---")
    cuda_status = "CUDA available" if torch.cuda.is_available() else "CUDA not available"
    gpu_count_str = f"Found {torch.cuda.device_count()} GPU(s)." if torch.cuda.is_available() else ""

    print(f"{cuda_status} {gpu_count_str}")
    print(f"Available devices detected by PyTorch: {', '.join(AVAILABLE_DEVICES)}")
    print(f"Default device selected by app: {DEFAULT_DEVICE}")
    # MODELS_DIR (which is "checkpoints") is used here
    print(f"Models from Styles (if Hub IDs) and additional local models will be loaded from/cached to: {os.path.abspath(MODELS_DIR)}")

    if not model_choices:
         print(f"\n!!! WARNING: No models available. The Gradio app will launch but cannot generate images.")
         print(f"Please define Styles and their corresponding Hub models in STYLE_MODEL_MAP in main.py")
         print(f"or add additional local diffusers models to '{MODELS_DIR}'. !!!") # Uses MODELS_DIR
    else:
         num_styled = len(STYLE_MODEL_MAP)
         num_additional_local = len(local_models_list) # Use the renamed list
         print(f"Defined {num_styled} featured style(s) mapping to Hub models.")
         print(f"Found {num_additional_local} additional local model(s) in '{os.path.abspath(MODELS_DIR)}'.") # Uses MODELS_DIR

    print("Launching Gradio interface...")
    demo.launch(show_error=True, inbrowser=True)
    print("Gradio interface closed.")
