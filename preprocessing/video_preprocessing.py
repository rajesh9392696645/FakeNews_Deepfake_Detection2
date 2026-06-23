import cv2
import numpy as np

FRAME_SIZE = (224, 224)

def extract_frames(video_path, max_frames=30):
    """
    Extract frames from video
    """

    cap = cv2.VideoCapture(video_path)

    frames = []

    while len(frames) < max_frames:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, FRAME_SIZE)

        frame = frame.astype("float32") / 255.0

        frames.append(frame)

    cap.release()

    return np.array(frames)


def sample_frames(video_path, sample_rate=10):
    """
    Extract every nth frame
    """

    cap = cv2.VideoCapture(video_path)

    frames = []

    frame_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % sample_rate == 0:

            frame = cv2.resize(frame, FRAME_SIZE)

            frame = frame.astype("float32") / 255.0

            frames.append(frame)

        frame_count += 1

    cap.release()

    return np.array(frames)


def preprocess_video(video_path):
    """
    Complete video preprocessing
    """

    return extract_frames(video_path)