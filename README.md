
# Computed Tomography Images for Intracranial Hemorrhage Detection and Segmentation

This project leverages Vision Transformers (ViT) and diffusion models to analyze computed tomography (CT) images for the detection and segmentation of intracranial hemorrhages. The aim is to extract meaningful features from CT images and generate visualizations that can assist in medical diagnosis.

## Overview

Intracranial hemorrhages are serious medical conditions that require immediate diagnosis and treatment. This project utilizes state-of-the-art deep learning techniques to process CT images, extract features, and create abstract visualizations that aid in understanding these medical images.

## Features

- **Image Processing**: Load NIfTI (.nii) format CT images and convert them into RGB format for feature extraction.
- **Feature Extraction**: Use the Vision Transformer model to extract high-level features from the images.
- **Image Generation**: Generate abstract visualizations based on the extracted features using a diffusion model.

## Getting Started

### Prerequisites

To run this project, you will need:

- Python 3.x
- Required libraries: `torch`, `transformers`, `nibabel`, `PIL`, `diffusers`, `safetensors`, `huggingface_hub`

You can install the required libraries using pip:

```bash
pip install torch torchvision transformers nibabel pillow diffusers safetensors huggingface_hub
```

### Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/CT_Hemorrhage_Detection_Segmentation.git
   ```

2. Navigate to the project directory:
   ```bash
   cd CT_Hemorrhage_Detection_Segmentation
   ```

3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook CT_Hemorrhage_Detection_and_Segmentation.ipynb
   ```

### Model and Data

- The project uses the **ViT (Vision Transformer)** model for feature extraction.
- **Stable Diffusion** is utilized for generating abstract visualizations.
- CT images can be found in the designated directory within the project structure.

## Results

The generated images based on the CT scans will be saved in the project directory. These images can serve as abstract representations for the respective CT images.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Google Vision Transformers](https://github.com/google-research/vision_transformer)
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [Nibabel](https://nipy.org/nibabel/)

For any questions or contributions, feel free to open an issue or a pull request.
