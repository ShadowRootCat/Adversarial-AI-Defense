import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Load the pre-trained MobileNetV2 model
mobileNetV2Model = tf.keras.applications.MobileNetV2(include_top=True, weights='imagenet')
mobileNetV2Model.trainable = False

def imagePreprocessor(inputImage):
    inputImage = tf.cast(inputImage, tf.float32)
    inputImage = tf.image.resize(inputImage, (224, 224))
    inputImage = tf.keras.applications.mobilenet_v2.preprocess_input(inputImage)
    inputImage = inputImage[None, ...]
    return inputImage

def adversarialPattern(inputImage, inputLabel):
    lossObject = tf.keras.losses.CategoricalCrossentropy()
    with tf.GradientTape() as tape:
        tape.watch(inputImage)
        prediction = mobileNetV2Model(inputImage)
        loss = lossObject(inputLabel, prediction)
    gradient = tape.gradient(loss, inputImage)
    signedGrad = tf.sign(gradient)
    return signedGrad

def imageOutput(images, descriptions):
    plt.figure(figsize=(len(images) * 3, 3))
    for i, image in enumerate(images):
        prediction = mobileNetV2Model.predict(image)
        label = tf.keras.applications.mobilenet_v2.decode_predictions(prediction, top=1)[0][0]
        plt.subplot(1, len(images), i + 1)
        plt.imshow(image[0] * 0.5 + 0.5)
        plt.title(f"{descriptions[i]}\n{label[1]}: {label[2]:.3f}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Example images
collectionOfImages = [
    ['plane.jpg', 'https://storage.googleapis.com/kagglesdsdata/datasets/4358290/7486255/vehicle_data/airplane/0011.jpg'],
    ['FerrariF1.jpg', 'https://storage.googleapis.com/kagglesdsdata/datasets/4358290/7486255/vehicle_data/car/Ferrari.jpg']
]

for testImage in collectionOfImages:
    decodedImage = tf.image.decode_image(tf.io.read_file(tf.keras.utils.get_file(testImage[0], testImage[1])))
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
