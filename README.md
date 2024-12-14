# Adversarial-AI-Defense
Adversarial AI Defense: A Cybersecurity Approach to Image Protection

# Adversarial Perturbation on Image Classifications

This project demonstrates the generation of adversarial perturbations on images using a pre-trained MobileNetV2 model. It highlights how small changes to input data can mislead machine learning models, showcasing a critical vulnerability in AI systems.

---

## Features

1. **Adversarial Example Generation**:

   - Generates adversarial perturbations using gradient-based methods.
   - Allows experimentation with different perturbation magnitudes (`epsilon`).

2. **Visualization**:

   - Displays original and perturbed images side-by-side.
   - Shows changes in classification confidence as perturbations are applied.

3. **Pre-trained Model**:

   - Utilizes MobileNetV2, pre-trained on the ImageNet dataset.

4. **Customizable**:

   - Easily adapt the code for different models or datasets.

---

## Purpose

This project serves as a practical exploration of adversarial machine learning concepts and their implications for AI security. It aims to:

- Highlight vulnerabilities in image classification models.
- Demonstrate the use of adversarial examples in testing model robustness.
- Educate users about the importance of adversarial defenses in AI systems.

---

## Installation

### Prerequisites

1. Python 3.8 or higher.

2. Install the required Python libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the pre-trained MobileNetV2 weights (handled automatically by TensorFlow).

---

## Usage

1. **Run the Script**:

   ```bash
   python perturbation_script.py
   ```

2. **Test Images**:

   - Use the provided test images in the `example_images/` directory.
   - Alternatively, update the `collectionOfImages` variable with new image URLs.

3. **Outputs**:

   - Displays original and perturbed images.
   - Outputs classification labels and confidence scores for both original and adversarial images.

---

## How it Works

1. **Pre-trained Model**:

   - Uses MobileNetV2 for classifying input images into one of 1,000 ImageNet categories.

2. **Adversarial Perturbation**:

   - Calculates the gradient of the loss with respect to the input image.
   - Generates perturbations based on the sign of the gradient (`sign(∇Loss)`).
   - Combines perturbations with the original image, scaled by a factor `epsilon`.

3. **Visualization**:

   - Displays how classification confidence changes as perturbations increase.

---



## Requirements

```plaintext
tensorflow
matplotlib
numpy
```

---

## Python Script: `perturbation_script.py`

---
