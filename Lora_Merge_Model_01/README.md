# 🚀 Whisper LoRA Adapter Merger

This repository contains a Python script designed to merge a PEFT (Parameter-Efficient Fine-Tuning) **LoRA (Low-Rank Adaptation)** adapter into a base Whisper model. The primary goal of this script is to take the lightweight, trained adapter weights and integrate them directly into the base model's weights.

The output is a new, standalone, and self-contained model that can be used for inference just like any regular `transformers` model, without requiring the `peft` library at runtime.

---

## 🤔 Why Merge a LoRA Adapter?

Training with LoRA is highly efficient because it only updates a small number of parameters (the adapter). However, for deployment and inference, keeping the adapter separate adds complexity. Merging provides several key benefits:

1.  **Deployment Simplicity**: Instead of loading a base model and then attaching a LoRA adapter, you deploy a single, consolidated model. This simplifies the inference pipeline and reduces dependencies.
2.  **No Performance Overhead**: During inference, the merged model has no computational overhead compared to the original base model architecture. There's no need to combine weights on the fly.
3.  **Model Portability**: The merged model is a standard `transformers` model. It can be easily shared, uploaded to the Hugging Face Hub, and used by anyone without needing the original LoRA adapter files or the `peft` library for inference.

---

## ✨ Features

-   **Seamless LoRA Merging**: Utilizes the `peft` library's `merge_and_unload` method to safely combine adapter and base model weights.
-   **Whisper Compatibility**: Includes a custom `WhisperPeftModel` class to correctly handle Whisper's unique `input_features` argument, preventing potential errors during model operations.
-   **Automated Saving**: Saves the newly merged model and its processor to a specified output directory, creating a complete, production-ready model package.
-   **Built-in Verification**: After saving, the script automatically loads the new model to verify its integrity and prints its structure and parameter count.
-   **Device Agnostic**: Uses `device_map="auto"` to automatically leverage available hardware (like GPUs) for efficient loading and processing.

---

## ⚙️ How It Works

The script executes the following steps:

1.  **Define Custom PEFT Class**: A `WhisperPeftModel` class is defined to override the default `forward` method. This is a crucial step to ensure that the `input_features` argument, specific to speech models like Whisper, is correctly passed to the base model.
2.  **Load Base Model**: It first loads a pre-existing Whisper model (which may already be fine-tuned) from the `merged_model_path`.
3.  **Load LoRA Adapter**: It then loads the LoRA adapter from a specific training checkpoint (`checkpoint_path`) and applies it to the base model, creating a `PeftModel` object.
4.  **Apply Custom Class**: The class of the `peft_model` object is dynamically switched to our custom `WhisperPeftModel`. This "monkey-patching" ensures full compatibility.
5.  **Merge Weights**: The core operation `peft_model.merge_and_unload()` is called. This function mathematically combines the adapter weights into the base model's weights and then frees the memory used by the adapter.
6.  **Save Artifacts**: The resulting `merged_model` (a standard `transformers` model) and its associated `processor` are saved to the `final_output_dir`.
7.  **Verify**: The script loads the model it just saved to confirm that the process was successful and the model is usable.

---

## 🔧 Configuration

To use this script, you need to set three main paths at the top of the file:

```python
# 1. Path to the base model that the LoRA adapter was trained on.
merged_model_path = "/path/to/your/base_whisper_model/"

# 2. Path to the directory containing the LoRA adapter checkpoint.
#    This is typically a 'checkpoint-XXXX' folder from your training run.
checkpoint_path = "/path/to/your/lora_adapter_checkpoint/"

# 3. Path where the new, fully merged model will be saved.
final_output_dir = "/path/to/save/new_merged_model/"
```

---

## 📦 Prerequisites & Installation

Ensure you have the necessary libraries installed. The `accelerate` library is important for handling `device_map="auto"`.

```bash
pip install torch transformers peft accelerate
```
*For GPU support, make sure your PyTorch installation is compatible with your CUDA version.*

---

## ▶️ Usage

1.  Update the three path variables (`merged_model_path`, `checkpoint_path`, `final_output_dir`) in the script.
2.  Run the script from your terminal:

    ```bash
    python your_merge_script.py
    ```

The script will log its progress to the console, and upon completion, the new merged model will be available in the specified output directory.

---

## 📄 Output

The script's primary output is a **new model directory** saved at `final_output_dir`. This directory will contain all the necessary files for a standard Hugging Face model, including:
-   `config.json`
-   `pytorch_model.bin` (or sharded `.safetensors` files)
-   `preprocessor_config.json`
-   `tokenizer.json` and other tokenizer files.

You will also see a confirmation message and the model's structure printed to the console at the end of the run.