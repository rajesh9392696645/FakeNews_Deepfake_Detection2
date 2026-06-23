from PIL import Image
import numpy as np
import os

OUTPUT_DIR = "generated_images"

def generate_fake_image():

    image = np.random.randint(
        0,
        255,
        (256,256,3),
        dtype=np.uint8
    )

    image = Image.fromarray(image)

    path = os.path.join(
        OUTPUT_DIR,
        "generated_image.png"
    )

    image.save(path)

    return path