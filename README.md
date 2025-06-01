# Perchance Revival - Recreating the Old Perchance SD 1.5 Experience

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Welcome to **Perchance Revival**! This user-friendly Gradio web application is specifically designed to replicate the experience of the old Perchance image generator by featuring the specific Stable Diffusion 1.5 models and common generation parameters/presets it used.

Generate images locally on your own PC (CPU or GPU) with a simple interface focused on providing that nostalgic Perchance feel. While the underlying application *can* technically load other SD1.5 models, its primary purpose and featured content are centered around the models and settings familiar to former Perchance users.

This project is designed for **Windows** users seeking a simple experience through easy-to-use batch files, as well as providing manual setup options for other platforms or advanced users.

## Application Screenshot:

![Screenshot of the Perchance Revival Web UI](images/ciphercore01.png)
*(Note: The screenshot shows the original UI design, which focuses on core SD1.5 generation. The 'Perchance Revival' version maintains this structure but highlights the specific models and likely includes presets within the UI itself)*

## ✨ Features

*   **Perchance Model & Preset Focus:** Access the specific Stable Diffusion 1.5 models and common generation parameters/presets used by the old Perchance website directly within the app. These models are downloaded and cached locally on first use.
*   **Flexible Model Loading:** *Beyond* the featured Perchance models, you can also load your own *additional* Stable Diffusion 1.5 models (in `diffusers` format) from a local `./checkpoints` directory or select other popular models from the Hugging Face Hub (as the underlying framework supports this).
*   **Device Agnostic:**
    *   Run inference on your **CPU**. (inference time around 4:55 with 10th gen i5)
    *   Leverage your **NVIDIA GPU** for significantly faster (Euler 30steps = 8 secs with 6GBVRAM) generation (requires installing the CUDA-enabled PyTorch version). **The default setup installs the CPU version; instructions are provided to upgrade for GPU users.**
*   **Comprehensive Control:**
    *   **Positive & Negative Prompts:** Guide the AI with detailed descriptions of what you want (and don't want).
    *   **Inference Steps:** Control the number of denoising steps.
    *   **CFG Scale:** Adjust how strongly the image should conform to your prompt.
    *   **Schedulers:** Experiment with different sampling algorithms (Euler, DPM++ 2M, DDPM, LMS).
    *   **Image Sizes:** Choose from standard SD1.5 resolutions, plus a "hire.fix" option (interpreted as 1024x1024).
    *   **Seed Control:** Set a specific seed for reproducible results or use -1 for random generation.
*   **User-Friendly Interface:**
    *   Clean and intuitive Gradio UI.
    *   Organized controls with advanced settings in an accordion for a cleaner look.
    *   Direct image display with download and share options.
*   **Safety First (Note):** The built-in safety checker is **disabled** in this version to allow for maximum creative freedom, aligning with the flexibility often sought in community generators like the old Perchance. Please be mindful of the content you generate.

## 🚀 Prerequisites

*   **Windows Operating System:** The provided batch files (`.bat`) are for Windows. For other operating systems, follow the manual setup steps below.
*   **Python:** 3.8 or higher. Ensure Python is installed and added to your system's PATH (usually an option during installation). You can download Python from [https://www.python.org/downloads/windows/](https://www.python.org/downloads/windows/).
*   **Git:** (Required for manual setup and updating) For cloning the repository.
*   **Hardware:**
    *   A modern CPU is required.
    *   For GPU acceleration (optional but highly recommended for speed), a compatible NVIDIA GPU with up-to-date CUDA drivers. At least 6-8GB VRAM is recommended for 512x512 generation, more for larger sizes.
*   **Internet Connection:** Required for downloading models from Hugging Face Hub (including the featured Perchance models) and for updates.

## 📦 Easy Setup (Windows - Download & Run)

This is the recommended method for most Windows users.

1.  **Download the project:**
    *   Go to the GitHub repository page: `[Insert your GitHub URL here, e.g., https://github.com/YourUsername/Perchance-Revival]`
    *   Click the green "Code" button.
    *   Click "Download ZIP".
2.  **Extract the ZIP:** Extract the downloaded ZIP file to a location on your computer (e.g., your Documents folder or Desktop). This will likely create a folder named something like `Perchance-Revival-main` (or similar). Rename it if you prefer, for example, to `PerchanceRevival`.
3.  **Run the Setup Script:**
    *   Navigate into the extracted folder (e.g., `PerchanceRevival`).
    *   Find the file named `setup.bat`.
    *   **Double-click `setup.bat`** to run it.
    *   A command prompt window will open. Follow the instructions in the window. This script will create a Python virtual environment (`venv`), install all necessary core dependencies, and install the **CPU version** of PyTorch by default.
    *   **Important:** Read the output in the command prompt carefully during and after the script finishes. It will provide specific commands if you wish to upgrade PyTorch to the GPU-accelerated CUDA version, which is necessary for fast generation on an NVIDIA GPU. You must run this upgrade command manually if you have a GPU and want to use it.
4.  **Prepare Additional Local Models (Optional):**
    *   Inside the extracted project folder (e.g., `PerchanceRevival`), create a directory named `checkpoints` (if `setup.bat` didn't create it).
    *   Place any *additional* Stable Diffusion 1.5 models (in `diffusers` format – meaning each model is a folder containing files like `model_index.json`, `unet/`, `vae/`, etc.) that you want to use *beyond* the featured Perchance models inside the `checkpoints` directory. The app will detect and list these alongside the default options.
        Example structure:
        ```
        PerchanceRevival/
        ├── checkpoints/          <-- For YOUR additional models
        │   ├── my-custom-model/
        │   │   ├── model_index.json
        │   │   ├── unet/
        │   │   └── ...
        │   └── another-local-model/
        │       └── ...
        ├── main.py              <-- Main application script
        ├── requirements.txt     <-- Dependency list
        ├── setup.bat            <-- Easy setup script
        ├── run.bat              <-- Easy run script
        ├── update.bat           <-- Easy update script
        ├── images/              <-- Folder containing example image
        │   └── ciphercore01.png <-- Example screenshot image file
        └── ...                  <-- Other app files
        ```

## 🔄 Updating the Application (Windows - Easy Method)

To get the latest code, dependency updates and updated models logic from this repository after using the easy setup:

*   Navigate to the project folder (e.g., `PerchanceRevival`).
*   Find the file named `update.bat`.
*   **Double-click `update.bat`** to run it.
*   A command prompt window will open and pull the latest changes from the GitHub repository and upgrade the Python packages in your virtual environment.
*   **Important:** This assumes you have not made local changes that conflict with the repository updates. If `git pull` fails, you may need to handle merge conflicts manually or discard local changes.


## ▶️ Running the Application (Windows - Easy Method)

Once the setup is complete, launch the Gradio web UI by double-clicking the `run.bat` file in your project folder (e.g., `PerchanceRevival`).

*   A command prompt window will open, activate the environment, and start the application.
*   A browser window should automatically open to the application (or a local URL will be provided in the console, usually `http://127.0.0.1:7860`).


---

## ⚙️ Manual Setup (Windows - Git Clone)

This method is for Windows users who are comfortable with Git.

1.  **Clone the Repository:** Open Command Prompt or PowerShell, navigate to where you want to download the project, and run:
    ```bash
    git clone [Insert your GitHub URL here, e.g., https://github.com/YourUsername/Perchance-Revival].git
    cd Perchance-Revival
    ```
    *(Note: If you cloned to a different directory name, replace `Perchance-Revival` above with your chosen directory name.)*
2.  **Proceed with Batch Files:** Continue by following **Step 2 (Run the Setup Script)**, **Step 4 (Prepare Additional Local Models)** (for your *own* checkpoints), **Running**, and **Updating** instructions from the **📦 Easy Setup (Windows - Download & Run)** section above. Make sure the `images` folder exists and contains `ciphercore01.png` as shown in the structure example if you use this method and they aren't already in the cloned repo.

## 🛠️ Manual Setup, Running & Updating (For Linux/macOS or Advanced Users)

If you are not on Windows or prefer a manual command-line approach:

1.  **Clone the Repository:**
    ```bash
    git clone [Insert your GitHub URL here, e.g., https://github.com/YourUsername/Perchance-Revival].git
    cd Perchance-Revival
    ```
2.  **Create and Activate a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
3.  **Install Dependencies (including PyTorch):**
    *   Install core dependencies (this includes `gradio`, `diffusers`, `transformers`, `huggingface_hub`, `Pillow`):
        ```bash
        pip install -r requirements.txt
        ```
    *   Install PyTorch: **This step is crucial and depends on your hardware.**
        *   **For CPU ONLY:**
            ```bash
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
            ```
        *   **For NVIDIA GPU with CUDA (Recommended for speed):** Find the appropriate command for your CUDA version (check PyTorch's website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)). Example for CUDA 11.8:
            ```bash
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
            ```
4.  **Prepare Additional Local Models (Optional):** Follow Step 4 from the **📦 Easy Setup (Windows - Download & Run)** section above (the part about the `checkpoints` folder for *your own* models).
5.  **Run the Application:**
    ```bash
    python main.py
    ```
    Ensure your virtual environment is activated (`source venv/bin/activate`) before running this command.
6.  **Updating Manually:**
    *   Navigate to the project directory in your terminal (`cd Perchance-Revival`).
    *   Ensure your virtual environment is activated (`source venv/bin/activate`).
    *   Pull the latest code: `git pull`
    *   Update dependencies: `pip install -r requirements.txt --upgrade`
    *   Deactivate the environment: `deactivate`

## ⚙️ Uninstall:

1.  **Delete the main directory (folder) - this app is completely portable.**


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://opensource.org/licenses/MIT) file for details.
