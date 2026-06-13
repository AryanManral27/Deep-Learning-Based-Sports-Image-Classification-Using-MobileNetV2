import numpy as np
from tensorflow.image import resize

# List of all possible sport classes
classes = [
    "air hockey", "archery", "arm wrestling", "baseball", "basketball", "boxing",
    "bull riding", "bungee jumping", "cricket", "football", "formula 1 racing", "golf",
    "hammer throw", "hockey", "horse jumping", "horse racing", "ice climbing", "javelin",
    "judo", "motorcycle racing", "olympic wrestling", "polo", "rock climbing", "rugby",
    "ski jumping", "skydiving", "sumo wrestling", "surfing", "swimming", "table tennis",
    "tennis", "tug of war", "volleyball", "water polo", "weightlifting"
]


def predict_label(img, model):
    """
    Predict the sport label and confidence for a given image.
    Returns (label, confidence) where confidence is 0–100, or (None, 0) if uncertain.
    """
    resized_img = resize(img, (224, 224)).numpy().astype("float32")
    resized_img = resized_img / 255.0
    exp_img = np.expand_dims(resized_img, axis=0)

    y_prob = model.predict(exp_img, verbose=0)
    confidence = float(y_prob.max(axis=-1)[0]) * 100

    if confidence < 50:
        return None, confidence

    label = classes[y_prob.argmax(axis=-1)[0]].title()
    return label, confidence
