from tensorflow.keras.models import load_model
from tensorflow.keras.utils import image_dataset_from_directory
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report

# Load your test dataset
test_ds = image_dataset_from_directory(
    directory= r"C:\Users\HP\Documents\sports-classifications\test",
    labels="inferred",
    label_mode="int",
    batch_size=32,
    image_size=(224, 224),
    shuffle=False  
)

# Preprocessing (normalize images)
def process(image, label):
    image = tf.image.resize(image, (224, 224))
    image = image / 255.0
    return image, label

test_ds = test_ds.map(process)
test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# Load trained model
model = load_model(r"C:\Users\HP\Downloads\sports-image-classification-main\streamlit\best_model.h5")

# Predict
y_pred = np.array([])
y_true = np.array([])

for x, y in test_ds:
    preds = np.argmax(model.predict(x), axis=-1)  # model.predict_classes() is deprecated
    y_pred = np.concatenate([y_pred, preds])
    y_true = np.concatenate([y_true, y.numpy()])

# Print results
print("Classification Report:\n", classification_report(y_true, y_pred))
