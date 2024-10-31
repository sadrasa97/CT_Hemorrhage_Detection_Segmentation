
# CT Scan Image Generation and Feature Extraction with ViT

## Overview
This project implements a workflow for extracting features from CT scan images using Vision Transformers (ViT) and generating high-resolution CT scan images via a Diffusion Model. The generated images can be used for fine-tuning a ViT model, aiming to enhance its capabilities for detecting intracranial hemorrhages.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Results](#results)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Prerequisites
Ensure you have the following installed:
- Python 3.8 or higher
- PyTorch with CUDA support
- Hugging Face Transformers library
- Diffusers library
- Nibabel for NIfTI image handling
- Pillow for image processing
- NumPy for numerical operations
- Hugging Face Hub for model downloads

## Installation
You can install the required packages using pip. Run the following command in your terminal:

```bash
pip install torch torchvision transformers diffusers nibabel Pillow numpy huggingface_hub safetensors
```

## Usage
1. **Load the Jupyter Notebook**: Open the provided Jupyter Notebook file in your preferred environment (e.g., Jupyter Lab, Google Colab).
2. **Prepare Data**: Ensure that your NIfTI (.nii) files are placed in the specified directory within the notebook.
3. **Run All Cells**: Execute all cells sequentially. The notebook is structured to guide you through the process of loading data, extracting features, generating images, and fine-tuning the ViT model.
4. **Inspect Results**: After running the notebook, inspect the generated images and the training results saved in the specified output directories.

### Example Code
Within the notebook, the following code snippet demonstrates the extraction of features and generation of images:

```python
# Extract features for each CT scan slice for hemorrhage detection
for filename in os.listdir(ct_scan_dir):
    if filename.endswith('.nii'):
        nii_path = os.path.join(ct_scan_dir, filename)
        try:
            slice_image = load_nii_as_image(nii_path)
            img_tensor = preprocess_image(slice_image)
            features = extract_features_vit(vit_model, img_tensor)
            feature_vectors[filename] = features
            print(f"Extracted features for {filename}")
        except IndexError as e:
            print(f"Error processing {filename}: {e}")
```

## Data Preparation
The data used in this project consists of NIfTI files containing CT scan images. The notebook handles loading these images, extracting 2D slices, normalizing them, and converting them into a format suitable for input into the ViT model.

### Directory Structure
```
/path/to/your/project
│
├── ct_scan_image_generation.ipynb # Main Jupyter Notebook
├── computed_tomography_images/      # Directory containing NIfTI files
│   └── *.nii
├── gen/                              # Output directory for generated images
│   └── generated_*.png
└── results/                          # Output directory for model training results
```

## Model Training
The notebook includes a section for fine-tuning the Vision Transformer model using the generated images. Key parameters for training include:

- **Batch Size**: 4
- **Number of Epochs**: 3
- **Logging Directory**: `./logs`
- **Checkpoint Directory**: `./results`

### Training Code Example
The following snippet demonstrates how to set up the `Trainer` for fine-tuning:

```python
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_steps=10,
    logging_dir="./logs",
)

trainer = Trainer(
    model=vit_model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
```

## Results
After running the notebook, generated images will be saved in the `gen/` directory, while training results and logs will be stored in the `results/` directory for further analysis.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [Hugging Face](https://huggingface.co/) for providing advanced models and libraries.
- [Diffusers](https://github.com/huggingface/diffusers) for image generation techniques.
- [Nibabel](https://nipy.org/nibabel/) for managing NIfTI files effectively.

