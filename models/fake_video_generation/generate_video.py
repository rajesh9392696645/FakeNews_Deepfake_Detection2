import cv2
import numpy as np
import os

OUTPUT_DIR = "generated_fake_videos"

def generate_fake_video():

    filename = os.path.join(
        OUTPUT_DIR,
        "generated_video.mp4"
    )

    writer = cv2.VideoWriter(
        filename,
        cv2.VideoWriter_fourcc(*'mp4v'),
        20,
        (640,480)
    )

    for _ in range(100):

        frame = np.random.randint(
            0,
            255,
            (480,640,3),
            dtype=np.uint8
        )

        writer.write(frame)

    writer.release()

    return filename