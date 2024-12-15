# Adversarial-AI-Defense


---

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
   pip3 install -r requirements.txt
   ```

3. Download the pre-trained MobileNetV2 weights (handled automatically by TensorFlow).

---

## Usage

1. **Run the Script**:

   ```bash
   python3 perturbation_script.py
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

## Python Script: `perturbation_script.py`

---

Example for example_images/
Create a folder named example_images/ in your project directory and populate it with test images. You can use the following files and URLs for testing:

```
example_images/
├── plane.jpg       # Downloaded from: https://storage.googleapis.com/kagglesdsdata/datasets/4358290/7486255/vehicle_data/airplane/0011.jpg
├── FerrariF1.jpg   # Downloaded from: https://storage.googleapis.com/kagglesdsdata/datasets/4358290/7486255/vehicle_data/car/Ferrari.jpg
You can either manually download these files or let the script automatically download them when it runs.
```

Here’s the modified code snippet to load images from the `example_images/` directory:

```python
# Define the directory containing example images
EXAMPLE_IMAGES_DIR = "example_images"

# Replace URL-based collectionOfImages with local file names
collectionOfImages = [
    "plane.jpg",
    "FerrariF1.jpg"
]

# Updated image loading logic
for testImage in collectionOfImages:
    image_path = os.path.join(EXAMPLE_IMAGES_DIR, testImage)
    decodedImage = tf.image.decode_image(tf.io.read_file(image_path))
    preprocessImage = imagePreprocessor(decodedImage)
    imageProbability = mobileNetV2Model.predict(preprocessImage)

    label = tf.keras.applications.mobilenet_v2.decode_predictions(imageProbability, top=1)[0][0]
    print(f"Original Prediction: {label[1]} with confidence {label[2]:.2f}")

    epsilons = [0.01, 0.1, 0.2]
    perturbations = adversarialPattern(preprocessImage, tf.one_hot(208, imageProbability.shape[-1]))

    correctedImages = [preprocessImage]
    descriptions = ["Original"]
    for eps in epsilons:
        adv_x = preprocessImage + eps * perturbations
        adv_x = tf.clip_by_value(adv_x, -1, 1)
        correctedImages.append(adv_x)
        descriptions.append(f"Epsilon = {eps}")

    imageOutput(correctedImages, descriptions)
```

### Key Changes:
1. **Image Path Construction**:
   - Used `os.path.join(EXAMPLE_IMAGES_DIR, testImage)` to construct paths relative to the `example_images/` directory.

2. **Replaced URLs**:
   - Removed the URLs and used filenames directly.

### Requirements:
- Place the files (e.g., `plane.jpg`, `FerrariF1.jpg`) in the `example_images/` directory.




### Google Colab Testing

You can directly test this project using Google Colab. The Colab environment has all dependencies pre-installed and ready to run.

- Open the project in Google Colab by clicking the link below:

[**Run in Google Colab**](<https://colab.research.google.com/drive/1pCa8NhD1yUeQlV80cvdscWI97ZDPoAD7bYsK_?usp=sharing>)


### Advisory: Do Not Modify the Code

**Important**: The provided code and configurations have been carefully designed and tested to demonstrate adversarial perturbation techniques effectively. 

- **Do Not Modify**:
  - Avoid altering the core logic of the script.
  - Changes to the pre-trained model settings, loss functions, or gradient calculations may lead to unexpected results.

- **Usage Guidelines**:
  - Use the code as-is for educational and testing purposes.
  - Ensure all dependencies are installed and follow the setup instructions.

- **Customizations**:
  - If you wish to adapt the project for other datasets, models, or use cases, please proceed with caution and ensure proper understanding of adversarial machine learning concepts.

