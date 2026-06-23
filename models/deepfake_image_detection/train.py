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
    Dropout,
    BatchNormalization,
    Input
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)

# ==================================================
# CONFIGURATION
# ==================================================

DATASET_PATH = "datasets/deepfake_images"

REAL_PATH = os.path.join(DATASET_PATH, "real")
FAKE_PATH = os.path.join(DATASET_PATH, "fake")

MODEL_PATH = "saved_models/deepfake_image_model.keras"

METRICS_PATH = "saved_models/deepfake_image_metrics.pkl"

IMG_SIZE = 224

# ==================================================
# LOAD IMAGES
# ==================================================

def load_images():

    images = []
    labels = []

    valid_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    )

    print("\nLoading REAL Images...")

    real_count = 0

    for file in os.listdir(REAL_PATH):

        if not file.lower().endswith(valid_extensions):
            continue

        path = os.path.join(REAL_PATH, file)

        image = cv2.imread(path)

        if image is None:
            continue

        image = cv2.resize(
            image,
            (IMG_SIZE, IMG_SIZE)
        )

        image = image / 255.0

        images.append(image)
        labels.append(0)

        real_count += 1

    print(f"REAL Images Loaded : {real_count}")

    print("\nLoading FAKE Images...")

    fake_count = 0

    for file in os.listdir(FAKE_PATH):

        if not file.lower().endswith(valid_extensions):
            continue

        path = os.path.join(FAKE_PATH, file)

        image = cv2.imread(path)

        if image is None:
            continue

        image = cv2.resize(
            image,
            (IMG_SIZE, IMG_SIZE)
        )

        image = image / 255.0

        images.append(image)
        labels.append(1)

        fake_count += 1

    print(f"FAKE Images Loaded : {fake_count}")

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

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # ==================================================
    # DATA AUGMENTATION
    # ==================================================

    datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.20,
        width_shift_range=0.20,
        height_shift_range=0.20,
        horizontal_flip=True
    )

    # ==================================================
    # CNN MODEL
    # ==================================================

    model = Sequential([

        Input(shape=(224,224,3)),

        Conv2D(
            32,
            (3,3),
            activation="relu"
        ),
        BatchNormalization(),
        MaxPooling2D((2,2)),

        Conv2D(
            64,
            (3,3),
            activation="relu"
        ),
        BatchNormalization(),
        MaxPooling2D((2,2)),

        Conv2D(
            128,
            (3,3),
            activation="relu"
        ),
        BatchNormalization(),
        MaxPooling2D((2,2)),

        Conv2D(
            256,
            (3,3),
            activation="relu"
        ),
        BatchNormalization(),
        MaxPooling2D((2,2)),

        Flatten(),

        Dense(
            256,
            activation="relu"
        ),

        Dropout(0.5),

        Dense(
            128,
            activation="relu"
        ),

        Dropout(0.3),

        Dense(
            1,
            activation="sigmoid"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # ==================================================
    # CALLBACKS
    # ==================================================

    os.makedirs(
        "saved_models",
        exist_ok=True
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
    )

    print("\nTraining Started...\n")

    model.fit(
        datagen.flow(
            X_train,
            y_train,
            batch_size=32
        ),
        epochs=25,
        validation_data=(
            X_test,
            y_test
        ),
        callbacks=[
            early_stop,
            checkpoint
        ]
    )

    print("\nEvaluating Model...\n")

    probabilities = model.predict(
        X_test,
        verbose=0
    )

    predicted_labels = (
        probabilities > 0.5
    ).astype(int).flatten()

    accuracy = accuracy_score(
        y_test,
        predicted_labels
    )

    precision = precision_score(
        y_test,
        predicted_labels,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predicted_labels,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predicted_labels,
        zero_division=0
    )

    print("=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)

    print(f"Accuracy  : {accuracy*100:.2f}%")
    print(f"Precision : {precision*100:.2f}%")
    print(f"Recall    : {recall*100:.2f}%")
    print(f"F1 Score  : {f1*100:.2f}%")

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

    metrics = {
        "accuracy": round(
            accuracy * 100,
            2
        ),
        "precision": round(
            precision * 100,
            2
        ),
        "recall": round(
            recall * 100,
            2
        ),
        "f1_score": round(
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

# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":
    train_model()
