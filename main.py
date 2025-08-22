# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 2024

@author: raxephion
Perchance Revival - Recreating Old Perchance SD 1.5 Experience
Basic Stable Diffusion 1.5 Gradio App with local/Hub models and CPU/GPU selection
Added multi-image generation, Hires. fix, Image-to-Image, and Secondary Styles with Perchance syntax parsing.
Models from STYLE_MODEL_MAP (if Hub IDs) will now be downloaded to MODELS_DIR.

NOTE: App still in early development - UI will be adjusted to match Perchance presets
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
import re # --- NEW: Import for regular expressions (parsing) ---
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

# --- Secondary Style Prompt Snippets ---
SECONDARY_STYLE_MAP = {
    "MTG Card": "magic the gathering",

"Realistic images": "high quality image, 4k, 8k, HD, UHD, Sharp Focus, In Frame",

"Realistic humans": "high quality image, 4k, 8k, HD, UHD",

"𝗡𝗼 𝘀𝘁𝘆𝗹𝗲": "",

"Anti-NSFW": "SFW",

"League of Legends": "fyptt.toconcept art, digital art, illustration, (league of legends style concept art), inspired by wlop style, 8k, fine details, sharp, very detailed, high resolution,anime, (realistic) ,magic the gathering, colorful background, no watermark,wallpaper, normal eyes",

"Jester": "(((jester laughing maniacally anime art style))) (((jester anime art style))) (((pointy jester hat anime art style))) (((yugioh anime art style))) (((neon colors))) (((vibrant colors))) (((bright colors))) (((lens flare))) (((light leaks))) (((long exposure)))",

"Ninja": "one character, ninja gaiden anime art style, ninja scroll anime art style, martial arts anime art style, 3D anime art style, heavy outlines art style, light leaks, lens flare, long exposure",

"Random Girl 1": "one character, female in her early 20s, {dark|light|medium} skin complexion, smooth skin, american face, {dark|light|medium} {blue|green|gray|hazel|brown} eyes, pretty lips, pretty eyes, light makeup, abstract halftone background, thick border around image, vibrant colors, bright colors, high contrast, {amazingly beautiful|embodiment of perfection|stunningly gorgeous} girl anime art style",

"Random Girl 2": "one character, female in her early 20s, {dark|light|medium} skin complexion, smooth skin, american face, {dark|light|medium} {blue|green|gray|hazel|brown} eyes, pretty lips, pretty eyes, light makeup, wearing {jeans|short shorts|a revealing outfit|a skin-tight bodysuit|a punk rock outfit|a steampunk outfit|a college cheerleader uniform|a skater girl outfit|a swimsuit|a bikini|underwear and a t-shirt|fancy underwear|a minidress with stockings|a miniskirt with stockings|leggings}, view from {the front|behind, rear projection}, {character portrait|{high-angle|low-angle|close up|over-the-shoulder|wide-angle|profile|full body|telephoto|panoramic|handheld|pov} shot|pov shot}, abstract halftone background, thick border around image, vibrant colors, bright colors, high contrast, {amazingly beautiful|embodiment of perfection|stunningly gorgeous} girl anime art style",

"Lego": "(legos art style:1.3), (lego video game art style:1.3)",

"Skittles": "(skittles art style:1.3), (taste the rainbow art style:1.3), skittles, tropical skittles, sour skittles, neon color grading, bright color grading, vibrant color grading, light leaks, lens flare, long exposure",

"Webcore": "(bold colors made in MS Paint and retro graphic design art style:1.3), (pixel art style:1.3), (neon color grading:1.2)",

"Terraria": "(Terrarria Art Style:1.3), (Pixel Art Style:1.3), (Vibrant Color Grading:1.3)",

"Final Fantasy": "(Final Fantasy Art Style:1.3), (CGI Video Game Art Style:1.3), (3D Video Game Art Style:1.3), (light leaks:1.1), (lens flare:1.1)",

"Star Wars Character": "(((one character:1.5))),(Star Wars Art Style:1.3), (Animated Star Wars Art style:1.3), (Star Wars Anime Art style:1.3), (Anime Art style:1.3), (bright color grading:1.2), (vibrant color grading:1.2), (light leaks), (lens flare)",

"Star Wars Battle": "(((epic space battle:1.5))),(Star Wars Space Battle Art Style:1.3), (Animated Star Wars Art style:1.3), (Space Battle Anime Art style:1.3), (Anime Art style:1.3), (Starship Battle Art Style:1.3), (bright color grading:1.2), (vibrant color grading:1.2), (light leaks), (lens flare)",

"Dragonball": "(Dragonball Anime Art Style:1.1), (YuGiOh art style:1.3), (bright color grading:1.2), (vibrant color grading:1.2), (light leaks), (lens flare)",

"Undertale": "(Undertale Anime Art Style:1.3)",


"ENA": " (AND NEW GAME created by Peruvian animator Joel Guerra art style:1.3), (Surrealist crossed with Late 90s and Early 2000s Computer software and obscure console gaming imagery art style:1.3), (bright color grading:1.2)",

"Neko (Catgirl)": "(((one character:1.5))), (human catgirl with a cat tail anime art style:1.3), (waifu anime art style:1.3), (gorgeous anime art style:1.3), (yugioh art style:1.3), (2d disney character art style:1.1), (catgirl anime art style:1.3), (neko anime art style:1.3), (vibrant color grading:1.6), (bright color grading:1.6), adult female catgirl, perfect body, {dark|light|medium} skin complexion, pretty lips, pretty eyes, light makeup, ({character portrait|{high-angle|low-angle|close up|over-the-shoulder|wide-angle|profile|full body|telephoto|panoramic|handheld} shot}:1.3)",

"American Girl": "(((one character:1.5))), (perfect gorgeous anime art style:1.3), (yugioh art style:1.3), (2d disney character art style:1.3), (gen13 comic art style), (stormwatch comic art style), tall adult female in her early 20s, perfect body, {dark|light|medium} skin complexion, smooth skin, american face, pretty lips, pretty eyes, light makeup, wearing {jeans|short shorts|a revealing outfit|a skin-tight bodysuit|a punk rock outfit|a steampunk outfit|a college cheerleader uniform|a skater girl outfit|a swimsuit|a bikini|underwear and a t-shirt with no bra|fancy underwear|a minidress with stockings|a miniskirt with stockings|leggings}:1.2, (view from {the front|behind}), ({character portrait|{high-angle|low-angle|close up|over-the-shoulder|wide-angle|profile|full body|telephoto|panoramic|handheld} shot}:1.3)",

"𝐍𝐒𝐅𝐖 - 𝐑𝐞𝐚𝐥𝐢𝐬𝐭𝐢𝐜": " highly realistic, realistic portrait, (nsfw), anatomically correct, realistic photograph, real colors, award winning photo, detailed face, realistic eyes, beautiful, sharp focus, high resolution, volumetric lighting, incredibly detailed, masterpiece, breathtaking, exquisite, great attention to skin and eyes",

"𝐍𝐒𝐅𝐖 - 𝐀𝐧𝐢𝐦𝐞": " intricate detail, hyper-anime, trending on artstation, 8k, fluid motion, stunning shading, anime, highly detailed, realistic, (nsfw), dramatic lighting, beautiful, animation, sharp focus, award winning, masterpiece, cinematic, dynamic, cinematic lighting, breathtaking, exquisite, great attention to skin and eyes, exceptional, exemplary, unsurpassed, viral, popular, buzzworthy, up-and-coming, emerging, promising, acclaimed, premium",

"𝐍𝐒𝐅𝐖 - 𝐑𝐞𝐚𝐥𝐢𝐬𝐭𝐢𝐜 (Stronger)": "highly realistic, realistic portrait, (((nsfw))), anatomically correct, realistic photograph, real colors, award winning photo, detailed face, realistic eyes, beautiful, sharp focus, high resolution, volumetric lighting, incredibly detailed, masterpiece, breathtaking, exquisite, great attention to skin and eyes",

"𝐍𝐒𝐅𝐖 - 𝐀𝐧𝐢𝐦𝐞 (Stronger)": " intricate detail, hyper-anime, trending on artstation, 8k, fluid motion, stunning shading, anime, highly detailed, realistic, (((nsfw))), dramatic lighting, beautiful, animation, sharp focus, award winning, masterpiece, cinematic, dynamic, cinematic lighting, breathtaking, exquisite, great attention to skin and eyes, exceptional, exemplary, unsurpassed, viral, popular, buzzworthy, up-and-coming, emerging, promising, acclaimed, premium",

"NSFW Painted Anime": "art by atey ghailan, painterly anime style at pixiv, art by kantoku, in art style of redjuice/necömi/rella/tiv pixiv collab, your name anime art style, masterpiece digital painting, exquisite lighting and composition, inspired by wlop art style, 8k, sharp, very detailed, high resolution, illustration ^2\n painterly anime artwork, [input.description], masterpiece, fine details, breathtaking artwork, painterly art style, high quality, 8k, very detailed, high resolution, exquisite composition and lighting, ((NSFW))",

"Realistic Human Generator": "in soft gaze, looking straight at the camera, skin blemishes, imperfect skin, skin pores, no makeup, no cosmetics, matured, solo, centered, RAW photo, detailed, clear features, sharp focus, film grain, 8k uhd, candid portrait, natural lighting",

"Teen": "(girl has babyface), ((first-year high-school)), (girl has flat chest), girl barely has boobs, girl is sooo adorable!!!, ((petite)), ((young)), teen, ((small boobs)), flat chest, tiny tits, (pink pussy), (young|virile|strong|chiseled|striking|nude|vagina|labia), shadowy, (shadows:1.3), (realistic vagina and labia), full body shot, torso, razor-sharp clarity, ultra-high resolution, photorealistic detail, natural color accuracy, perfectly balanced exposure, uncompromised texture fidelity, focus precision, real-world vibrancy, authentic ambient lighting, impeccable contrast levels, true-to-life saturation, crystal-clear sharpness, unaltered realness, fine-grained definition, optical purity, ambient light capture, shadow and highlight detail, unfiltered realism, (real:1.3), studio lighting, shiny skin, crazy bitch, real photograph, small, teen, teenager, young, (open skin:1.2), headshot, {nerdy glasses|Dental braces}, {Standing Straight|Crossed Legs|one Leg Forward|Hands on Hips|Leaning Against Something|Sitting Down|Kneeling|Lying Down|Back to Camera|Tiptoe Stand|Walking Pose|Jumping Pose|Action Pose|Arms Crossed|Hands in Pockets|Hand on Hip|Arms Above Head|Hands Together in Front|Reaching out|Playing with Hair|Touching Face|Holding an object|Prayer Position|Hands on Shoulders|Fingers to Lips|Waving|Looking over Shoulder|Tilting Head|Direct Gaze into Camera|Side Glance|Closed Eyes|Smiling|Serious Expression|Laughing|Blowing a Kiss|Winking|Eyes Wide open|Frowning|Surprised Look|one Foot Forward|Crossed at the Ankles|Bent Knee|Pointed Toes|Feet",

"Style of 'Your Name'": "in the art style of Makoto Shinkai, beautiful detailed skies, cinematic lighting, vibrant colors, emotional and atmospheric",

"Pokemon": "in the art style of Ken Sugimori, official Pokemon art, simple and bold outlines, vibrant and playful color palette, dynamic character poses",

"Berserk anime style": "in the art style of Kentaro Miura, dark fantasy, intricate and detailed linework, heavy shading and cross-hatching, dramatic and gritty atmosphere, monochrome with muted tones",

"Studio Ghibli": "in the style of Studio Ghibli and Hayao Miyazaki, whimsical and enchanting, lush hand-painted watercolor backgrounds, soft and nostalgic lighting, dreamlike fantasy",

"Cuphead": "in the style of a 1930s rubber hose animation, vintage cartoon, hand-drawn aesthetic, surreal and whimsical characters, grainy and aged film look",

"Psychedelic Art": "psychedelic style, vibrant swirling colors, abstract and surreal patterns, kaleidoscopic and fractal visuals, mind-bending and trippy",

"Into the Spider-Verse": "comic book art style, ben-day dots, chromatic aberration, halftone patterns, dynamic and energetic composition, vibrant and saturated colors, graphic and stylized",

"Vaporwave": "80s and 90s retro futurism, neon grids, pastel color palette (pinks, blues, purples), roman busts, glitch art, Japanese text, nostalgic and surreal",

"Tim Burton Style": "gothic and whimsical, dark yet playful, spindly and exaggerated character designs, large expressive eyes, muted color palette with occasional bursts of color, twisted and fantastical"
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

# --- NEW: Helper function to parse Perchance-style syntax ---
def parse_perchance_syntax(prompt_text):
    """
    Finds and replaces Perchance-style {a|b|c} syntax with a random choice.
    This function loops to handle nested syntax correctly.
    """
    pattern = re.compile(r'{([^{}]+?)}')
    while True:
        match = pattern.search(prompt_text)
        if not match:
            break
        
        options_str = match.group(1)
        choices = [choice.strip() for choice in options_str.split('|')]
        selected_choice = random.choice(choices)
        
        prompt_text = prompt_text[:match.start()] + selected_choice + prompt_text[match.end():]
        
    return prompt_text

# --- Model Loading Helper Function ---
def load_model(model_input_name, selected_device_str):
    global current_pipeline, current_model_id_loaded, current_device_loaded

    device_to_use = "cuda" if selected_device_str == "GPU" and "GPU" in AVAILABLE_DEVICES else "cpu"
    dtype_to_use = torch.float16 if device_to_use == 'cuda' else torch.float32
    actual_model_id_to_load = STYLE_MODEL_MAP.get(model_input_name, model_input_name)

    if not actual_model_id_to_load or actual_model_id_to_load == "No models found":
        raise gr.Error("Invalid model selection. Could not determine which model to load.")

    if current_pipeline is None or current_model_id_loaded != actual_model_id_to_load or str(current_device_loaded) != device_to_use:
        print(f"Loading model: {actual_model_id_to_load} onto {device_to_use}...")
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
            pipeline.to(device_to_use)
            current_pipeline = pipeline
            current_model_id_loaded = actual_model_id_to_load
            current_device_loaded = torch.device(device_to_use)
            print(f"Model '{actual_model_id_to_load}' loaded successfully.")
        except Exception as e:
            raise gr.Error(f"Failed to load model '{actual_model_id_to_load}': {e}. Check model name and internet connection.")

    return device_to_use

# --- Text-to-Image Generation Function ---
def generate_image_from_text(model_input_name, selected_device_str, secondary_style_name, prompt, negative_prompt, steps, cfg_scale, scheduler_name, size, seed, num_images,
                   hires_fix_enable, denoising_strength, upscale_by):
    if not prompt:
        raise gr.Error("Please enter a prompt.")

    if secondary_style_name and secondary_style_name != "None":
        style_prompt = SECONDARY_STYLE_MAP.get(secondary_style_name, "")
        if style_prompt:
            prompt = f"{prompt}, {style_prompt}"
            print(f"Appended secondary style '{secondary_style_name}'.")

    # --- MODIFIED: Parse the final prompt for Perchance syntax ---
    prompt = parse_perchance_syntax(prompt)
    print(f"Final Parsed Prompt: {prompt}")

    device_to_use = load_model(model_input_name, selected_device_str)
    
    selected_scheduler_class = SCHEDULER_MAP.get(scheduler_name, SCHEDULER_MAP[DEFAULT_SCHEDULER])
    current_pipeline.scheduler = selected_scheduler_class.from_config(current_pipeline.scheduler.config)

    try:
        w_str, h_str = size.split('x')
        width, height = int(w_str), int(h_str)
    except ValueError:
        raise gr.Error(f"Invalid size format: '{size}'. Use 'WidthxHeight'.")

    seed_int = int(seed)
    if seed_int == -1:
        seed_int = random.randint(0, MAX_SEED)
    generator = torch.Generator(device=device_to_use).manual_seed(seed_int)
    
    num_images_int = int(num_images)
    print(f"Generating {num_images_int} image(s) with seed {seed_int}...")
    start_time = time.time()

    try:
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

        if hires_fix_enable:
            print("Hires. fix enabled. Starting second pass...")
            pipe_img2img = StableDiffusionImg2ImgPipeline(**current_pipeline.components).to(device_to_use)
            hires_images = []
            for i, base_image in enumerate(base_images):
                new_width, new_height = int(base_image.width * upscale_by), int(base_image.height * upscale_by)
                upscaled_image = base_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                generator_hires = torch.Generator(device=device_to_use).manual_seed(seed_int + i)
                output_hires = pipe_img2img(prompt=prompt, negative_prompt=negative_prompt, image=upscaled_image, strength=denoising_strength, guidance_scale=float(cfg_scale), generator=generator_hires, num_inference_steps=int(steps)).images
                hires_images.extend(output_hires)
            generated_images_list = hires_images
        else:
            generated_images_list = base_images

        end_time = time.time()
        print(f"Generation finished in {end_time - start_time:.2f} seconds.")
        return generated_images_list, seed_int
    except Exception as e:
        raise gr.Error(f"An error occurred during generation: {e}")

# --- Image-to-Image Generation Function ---
def generate_image_from_image(model_input_name, selected_device_str, secondary_style_name, input_image, prompt, negative_prompt, strength, steps, cfg_scale, scheduler_name, seed, num_images):
    if input_image is None:
        raise gr.Error("Please upload an input image for Image-to-Image generation.")

    if secondary_style_name and secondary_style_name != "None":
        style_prompt = SECONDARY_STYLE_MAP.get(secondary_style_name, "")
        if style_prompt:
            prompt = f"{prompt}, {style_prompt}"
            print(f"Appended secondary style '{secondary_style_name}'.")

    # --- MODIFIED: Parse the final prompt for Perchance syntax ---
    prompt = parse_perchance_syntax(prompt)
    print(f"Final Parsed Prompt: {prompt}")

    device_to_use = load_model(model_input_name, selected_device_str)
    
    pipe_img2img = StableDiffusionImg2ImgPipeline(**current_pipeline.components).to(device_to_use)
    selected_scheduler_class = SCHEDULER_MAP.get(scheduler_name, SCHEDULER_MAP[DEFAULT_SCHEDULER])
    pipe_img2img.scheduler = selected_scheduler_class.from_config(pipe_img2img.scheduler.config)

    seed_int = int(seed)
    if seed_int == -1:
        seed_int = random.randint(0, MAX_SEED)
    generator = torch.Generator(device=device_to_use).manual_seed(seed_int)

    num_images_int = int(num_images)
    print(f"Generating {num_images_int} image(s) from image with seed {seed_int}...")
    
    input_image_pil = Image.fromarray(input_image).convert("RGB")
    
    start_time = time.time()
    try:
        output_images = pipe_img2img(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=input_image_pil,
            strength=float(strength),
            guidance_scale=float(cfg_scale),
            num_inference_steps=int(steps),
            generator=generator,
            num_images_per_prompt=num_images_int,
        ).images

        end_time = time.time()
        print(f"Generation finished in {end_time - start_time:.2f} seconds.")
        return output_images, seed_int
    except Exception as e:
        raise gr.Error(f"An error occurred during image-to-image generation: {e}")

# --- CSS and Theming ---
cyberpunk_css = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400..900&display=swap');
:root { --neon-blue: #00ffff; --neon-glow: 0 0 8px #00ffff, 0 0 16px #00ffff; --bg-black: #000000; }
body, .gradio-container { background: var(--bg-black) !important; color: var(--neon-blue) !important; font-family: 'Orbitron', sans-serif !important; }
#main_title { color: var(--neon-blue) !important; text-shadow: var(--neon-glow); text-align: center; font-size: 3em !important; border-bottom: 1px solid var(--neon-blue); padding-bottom: 8px; }
textarea, input[type="text"], input[type="number"], .gr-textbox, .gr-input, .gradio-dropdown { background: var(--bg-black) !important; border: 1px solid var(--neon-blue) !important; color: var(--neon-blue) !important; text-shadow: var(--neon-glow); font-family: 'Orbitron', sans-serif !important; border-radius: 0 !important; box-shadow: inset 0 0 12px #003333; }
.gradio-gallery, .gradio-image { border: 1px solid var(--neon-blue) !important; border-radius: 0 !important; background: #001111 !important; }
button, .gr-button { background: var(--bg-black) !important; border: 1px solid var(--neon-blue) !important; color: var(--neon-blue) !important; text-shadow: var(--neon-glow); border-radius: 0 !important; transition: all 0.2s ease-in-out; }
button:hover, .gr-button:hover { background: var(--neon-blue) !important; color: var(--bg-black) !important; text-shadow: none !important; box-shadow: 0 0 20px #00ffff; }
.gr-accordion, .gr-box, .gradio-tabs > .tab-nav > button { background: var(--bg-black) !important; border: 1px solid var(--neon-blue) !important; color: var(--neon-blue) !important; border-radius: 0 !important; }
.gradio-tabs > .tab-nav > button.selected { background: var(--neon-blue) !important; color: var(--bg-black) !important; text-shadow: none !important; }
input[type=range] { -webkit-appearance: none; background: transparent; }
input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; height: 16px; width: 16px; border-radius: 0; background: var(--neon-blue); cursor: pointer; margin-top: -7px; box-shadow: var(--neon-glow); }
input[type=range]::-webkit-slider-runnable-track { width: 100%; height: 2px; cursor: pointer; background: var(--neon-blue) !important; border: 1px solid var(--neon-blue); }
h3, label, .gr-checkbox label span { color: var(--neon-blue) !important; text-shadow: var(--neon-glow); font-weight: 600; text-transform: uppercase; }
"""

# --- Gradio Interface ---
styled_models = list(STYLE_MODEL_MAP.keys())
model_choices = styled_models if styled_models else ["No models found"]
initial_default_model = model_choices[0]
scheduler_choices = list(SCHEDULER_MAP.keys())
secondary_style_choices = ["None"] + list(SECONDARY_STYLE_MAP.keys())

with gr.Blocks(css=cyberpunk_css) as demo:
    gr.Markdown("# Perchance Revival", elem_id="main_title")

    with gr.Row():
        model_dropdown = gr.Dropdown(choices=model_choices, value=initial_default_model, label="Select Style", interactive=bool(styled_models), scale=3)
        secondary_style_dropdown = gr.Dropdown(choices=secondary_style_choices, value="None", label="Secondary Style (optional)", interactive=True, scale=2)
        device_dropdown = gr.Dropdown(choices=AVAILABLE_DEVICES, value=DEFAULT_DEVICE, label="Processing Device", interactive=len(AVAILABLE_DEVICES) > 1, scale=1)

    with gr.Tabs():
        # --- TEXT-TO-IMAGE TAB ---
        with gr.TabItem("Text-to-Image"):
            with gr.Row():
                with gr.Column(scale=2):
                    prompt_input = gr.Textbox(label="Positive Prompt", placeholder="Enter your prompt...", lines=3)
                    negative_prompt_input = gr.Textbox(label="Negative Prompt", placeholder="e.g. blurry, bad quality...", lines=2)
                    generate_button_txt2img = gr.Button("Generate", variant="primary")
                    with gr.Accordion("Advanced Settings", open=False):
                        steps_slider = gr.Slider(minimum=5, maximum=150, value=30, label="Inference Steps", step=1)
                        cfg_slider = gr.Slider(minimum=1.0, maximum=30.0, value=7.5, label="CFG Scale", step=0.1)
                        scheduler_dropdown = gr.Dropdown(choices=scheduler_choices, value=DEFAULT_SCHEDULER, label="Scheduler")
                        size_dropdown = gr.Dropdown(choices=SUPPORTED_SD15_SIZES, value="512x768", label="Image Size")
                        seed_input = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                        num_images_slider = gr.Slider(minimum=1, maximum=12, value=1, step=1, label="Number of Images")
                    with gr.Accordion("Hires. fix", open=False):
                        hires_fix_enable_checkbox = gr.Checkbox(label="Enable Hires. fix", value=False)
                        denoising_strength_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.01, label="Denoising Strength")
                        upscale_by_slider = gr.Slider(minimum=1.0, maximum=2.0, value=1.5, step=0.05, label="Upscale by")
                with gr.Column(scale=3):
                    output_gallery_txt2img = gr.Gallery(label="Generated Images", show_label=True, interactive=False, format="png", columns=4, object_fit="contain")
                    actual_seed_output_txt2img = gr.Number(label="Actual Seed Used", precision=0, interactive=False)
        
        # --- IMAGE-TO-IMAGE TAB ---
        with gr.TabItem("Image-to-Image"):
            with gr.Row():
                with gr.Column(scale=2):
                    input_image_img2img = gr.Image(type="numpy", label="Input Image")
                    strength_slider_img2img = gr.Slider(minimum=0.0, maximum=1.0, value=0.75, label="Strength", info="How much noise to add to the input image (0.0 preserves original, 1.0 is very creative)")
                    prompt_input_img2img = gr.Textbox(label="Positive Prompt", placeholder="Describe what you want to see...", lines=3)
                    negative_prompt_input_img2img = gr.Textbox(label="Negative Prompt", placeholder="e.g. blurry, bad quality...", lines=2)
                    generate_button_img2img = gr.Button(" Generate from Image ", variant="primary")
                    with gr.Accordion("Advanced Settings", open=False):
                        steps_slider_img2img = gr.Slider(minimum=5, maximum=150, value=30, label="Inference Steps", step=1)
                        cfg_slider_img2img = gr.Slider(minimum=1.0, maximum=30.0, value=7.5, label="CFG Scale", step=0.1)
                        scheduler_dropdown_img2img = gr.Dropdown(choices=scheduler_choices, value=DEFAULT_SCHEDULER, label="Scheduler")
                        seed_input_img2img = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                        num_images_slider_img2img = gr.Slider(minimum=1, maximum=9, value=1, step=1, label="Number of Images")
                with gr.Column(scale=3):
                    output_gallery_img2img = gr.Gallery(label="Generated Images", show_label=True, interactive=False, format="png", columns=4, object_fit="contain")
                    actual_seed_output_img2img = gr.Number(label="Actual Seed Used", precision=0, interactive=False)

    # --- Define Button Clicks ---
    generate_button_txt2img.click(
        fn=generate_image_from_text,
        inputs=[model_dropdown, device_dropdown, secondary_style_dropdown, prompt_input, negative_prompt_input, steps_slider, cfg_slider, scheduler_dropdown, size_dropdown, seed_input, num_images_slider, hires_fix_enable_checkbox, denoising_strength_slider, upscale_by_slider],
        outputs=[output_gallery_txt2img, actual_seed_output_txt2img]
    )
    generate_button_img2img.click(
        fn=generate_image_from_image,
        inputs=[model_dropdown, device_dropdown, secondary_style_dropdown, input_image_img2img, prompt_input_img2img, negative_prompt_input_img2img, strength_slider_img2img, steps_slider_img2img, cfg_slider_img2img, scheduler_dropdown_img2img, seed_input_img2img, num_images_slider_img2img],
        outputs=[output_gallery_img2img, actual_seed_output_img2img]
    )
    
    gr.Markdown(f"Models are downloaded to the `{MODELS_DIR}` folder.")

if __name__ == "__main__":
    print("--- Starting Perchance Revival ---")
    print(f"Models will be cached in: {os.path.abspath(MODELS_DIR)}")
    demo.launch(show_error=True, inbrowser=True)
    print("Gradio interface closed.")
