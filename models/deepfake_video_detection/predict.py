import numpy as np
import tensorflow as tf

MODEL_PATH = "saved_models/deepfake_video_model.h5"

model = tf.keras.models.load_model(
    MODEL_PATH
)

def predict_video(video_frames):

    prediction = model.predict(
        np.expand_dims(
            video_frames,
            axis=0
        )
    )

    confidence = float(
        prediction[0][0]
    )

    label = (
        "Fake"
        if confidence > 0.5
        else "Real"
    )

    return label, round(confidence*100,2)