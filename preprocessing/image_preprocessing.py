import cv2
import numpy as np

IMG_SIZE = (224, 224)

def load_image(image_path):
    """
    Load image from disk
    """

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Unable to load image")

    return image


def resize_image(image):
    """
    Resize image
    """

    return cv2.resize(image, IMG_SIZE)


def normalize_image(image):
    """
    Normalize image
    """

    image = image.astype("float32")

    image = image / 255.0

    return image


def preprocess_image(image_path):
    """
    Complete image preprocessing
    """

    image = load_image(image_path)

    image = resize_image(image)

    image = normalize_image(image)

    return image


def image_to_batch(image):
    """
    Convert image into model input batch
    """

    return np.expand_dims(image, axis=0)