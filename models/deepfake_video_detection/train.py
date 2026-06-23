import tensorflow as tf

MODEL_PATH = "saved_models/deepfake_video_model.h5"

def build_video_model():

    model = tf.keras.Sequential([

        tf.keras.layers.LSTM(
            64,
            input_shape=(30,224*224*3)
        ),

        tf.keras.layers.Dense(
            64,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            1,
            activation="sigmoid"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

if __name__ == "__main__":

    model = build_video_model()

    model.save(MODEL_PATH)

    print("Video Model Saved")