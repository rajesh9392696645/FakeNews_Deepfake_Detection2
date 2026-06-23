import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------------------------------
# Text Features (Fake News Detection)
# -------------------------------------------------

def extract_text_features(train_texts,
                          test_texts=None,
                          max_features=5000):

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english'
    )

    X_train = vectorizer.fit_transform(train_texts)

    if test_texts is not None:
        X_test = vectorizer.transform(test_texts)
        return X_train, X_test, vectorizer

    return X_train, vectorizer


# -------------------------------------------------
# Image Features
# -------------------------------------------------

def extract_image_features(image):

    image = image.flatten()

    return image


# -------------------------------------------------
# Video Features
# -------------------------------------------------

def extract_video_features(video_frames):

    if len(video_frames) == 0:
        return np.array([])

    features = np.mean(video_frames, axis=0)

    return features.flatten()


# -------------------------------------------------
# Combined Feature Extraction
# -------------------------------------------------

def get_feature_summary(data):

    return {
        "shape": data.shape,
        "dimensions": len(data.shape),
        "size": data.size
    }