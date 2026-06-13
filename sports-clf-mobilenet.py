from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
import tensorflow as tf
import numpy as np

# Load Dataset
train_ds = image_dataset_from_directory(
    directory=r"C:\Users\HP\Documents\sports-classifications\train",
    labels="inferred",
    label_mode="int",
    batch_size=32,
    image_size=(224, 224),
)

validation_ds = image_dataset_from_directory(
    directory=r"C:\Users\HP\Documents\sports-classifications\valid",
    labels="inferred",
    label_mode="int",
    batch_size=32,
    image_size=(224, 224),
)

test_ds = image_dataset_from_directory(
    directory=r"C:\Users\HP\Documents\sports-classifications\test",
    labels="inferred",
    label_mode="int",
    batch_size=32,
    image_size=(224, 224),
)

# Data Preprocessing + Augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.2),
])

def process(image, label):
    image = tf.image.resize(image, (224, 224))
    image = image / 255.0
    return image, label

train_ds = train_ds.map(process).map(lambda x, y: (data_augmentation(x, training=True), y))
validation_ds = validation_ds.map(process)
test_ds = test_ds.map(process)

train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
validation_ds = validation_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# Model (Using MobileNetV2 as Base)
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
base_model.trainable = False  # Freeze base layers initially

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(100, activation="softmax")  # Change '100' to your actual number of classes
])

# Compile & Train
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callback = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

history = model.fit(
    train_ds,
    epochs=20,
    validation_data=validation_ds,
    callbacks=[callback],
)

# Evaluate & Report
y_pred = np.array([])
y_true = np.array([])

for x, y in test_ds:
    preds = np.argmax(model.predict(x), axis=-1)
    y_pred = np.concatenate([y_pred, preds])
    y_true = np.concatenate([y_true, y.numpy()])

print("Classification Report:\n", classification_report(y_true, y_pred))

# Save Model
model.save("best_model.h5")
print("\nModel saved as: best_model.h5")
