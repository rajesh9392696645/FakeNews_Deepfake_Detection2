import os
import cv2
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)

from tensorflow.keras.utils import to_categorical

# ==================================================
# CONFIGURATION
# ==================================================

DATASET_PATH = "datasets/deepfake_images"

REAL_PATH = os.path.join(
    DATASET_PATH,
    "real"
)

FAKE_PATH = os.path.join(
    DATASET_PATH,
    "fake"
)

MODEL_PATH = (
    "saved_models/deepfake_image_model.h5"
)

METRICS_PATH = (
    "saved_models/deepfake_image_metrics.pkl"
)

IMG_SIZE = 224

# ==================================================
# LOAD DATASET
# ==================================================

def load_images():

    images = []
    labels = []

    print("\nLoading REAL Images...")

    for file in os.listdir(REAL_PATH):

        path = os.path.join(
            REAL_PATH,
            file
        )

        try:

            image = cv2.imread(path)

            image = cv2.resize(
                image,
                (IMG_SIZE, IMG_SIZE)
            )

            image = image / 255.0

            images.append(image)

            labels.append(0)

        except:
            continue

    print(
        f"REAL Images Loaded: {len(labels)}"
    )

    real_count = len(labels)

    print("\nLoading FAKE Images...")

    for file in os.listdir(FAKE_PATH):

        path = os.path.join(
            FAKE_PATH,
            file
        )

        try:

            image = cv2.imread(path)

            image = cv2.resize(
                image,
                (IMG_SIZE, IMG_SIZE)
            )

            image = image / 255.0

            images.append(image)

            labels.append(1)

        except:
            continue

    fake_count = len(labels) - real_count

    print(
        f"FAKE Images Loaded: {fake_count}"
    )

    return (
        np.array(images),
        np.array(labels)
    )

# ==================================================
# TRAIN MODEL
# ==================================================

def train_model():

    print("=" * 60)
    print("DEEPFAKE IMAGE DETECTION TRAINING")
    print("=" * 60)

    X, y = load_images()

    print("\nDataset Shape:")
    print(X.shape)

    print("\nSplitting Dataset...")

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )

    y_train_cat = to_categorical(
        y_train,
        num_classes=2
    )

    y_test_cat = to_categorical(
        y_test,
        num_classes=2
    )

    print("\nBuilding CNN Model...")

    model = Sequential()

    model.add(
        Conv2D(
            32,
            (3,3),
            activation="relu",
            input_shape=(224,224,3)
        )
    )

    model.add(
        MaxPooling2D((2,2))
    )

    model.add(
        Conv2D(
            64,
            (3,3),
            activation="relu"
        )
    )

    model.add(
        MaxPooling2D((2,2))
    )

    model.add(
        Conv2D(
            128,
            (3,3),
            activation="relu"
        )
    )

    model.add(
        MaxPooling2D((2,2))
    )

    model.add(
        Flatten()
    )

    model.add(
        Dense(
            128,
            activation="relu"
        )
    )

    model.add(
        Dropout(0.5)
    )

    model.add(
        Dense(
            2,
            activation="softmax"
        )
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\nTraining Started...")

    model.fit(
        X_train,
        y_train_cat,
        epochs=10,
        batch_size=32,
        validation_split=0.1
    )

    print("\nEvaluating Model...")

    predictions = model.predict(X_test)

    predicted_labels = np.argmax(
        predictions,
        axis=1
    )

    accuracy = accuracy_score(
        y_test,
        predicted_labels
    )

    precision = precision_score(
        y_test,
        predicted_labels
    )

    recall = recall_score(
        y_test,
        predicted_labels
    )

    f1 = f1_score(
        y_test,
        predicted_labels
    )

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)

    print(
        f"Accuracy  : {accuracy*100:.2f}%"
    )

    print(
        f"Precision : {precision*100:.2f}%"
    )

    print(
        f"Recall    : {recall*100:.2f}%"
    )

    print(
        f"F1 Score  : {f1*100:.2f}%"
    )

    print("\nClassification Report\n")

    print(
        classification_report(
            y_test,
            predicted_labels,
            target_names=[
                "REAL",
                "FAKE"
            ]
        )
    )

    os.makedirs(
        "saved_models",
        exist_ok=True
    )

    model.save(
        MODEL_PATH
    )

    metrics = {

        "accuracy":
            round(
                accuracy * 100,
                2
            ),

        "precision":
            round(
                precision * 100,
                2
            ),

        "recall":
            round(
                recall * 100,
                2
            ),

        "f1_score":
            round(
                f1 * 100,
                2
            )
    }

    joblib.dump(
        metrics,
        METRICS_PATH
    )

    print("\nModel Saved Successfully")

    print(MODEL_PATH)

    print("\nMetrics Saved Successfully")

    print(METRICS_PATH)


if __name__ == "__main__":

    train_model()