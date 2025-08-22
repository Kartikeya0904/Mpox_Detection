import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pathlib

# Define dataset paths
train_dir = pathlib.Path("C:/MPOX CODE/DATA_AFTER_SPLIT/aug_train")
val_dir = pathlib.Path("C:/MPOX CODE/DATA_AFTER_SPLIT/Valid")
test_dir = pathlib.Path("C:/MPOX CODE/DATA_AFTER_SPLIT/Test")

# Load disease mappings from CSV files
def load_disease_mapping(file_path):
    mapping_df = pd.read_csv(file_path)
    return dict(zip(mapping_df['Filename'], mapping_df['Disease']))

# Load mappings for train, val, and test datasets
train_mapping = load_disease_mapping("C:/MPOX CODE/train_disease_mapping.csv")
val_mapping = load_disease_mapping("C:/MPOX CODE/val_mapping.csv")
test_mapping = load_disease_mapping("C:/MPOX CODE/test_mapping.csv")

# Set parameters
batch_size = 32
img_height = 224
img_width = 224
epochs = 20
lr_rate = 1e-4

# Load datasets
train_ds = image_dataset_from_directory(
    train_dir,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True
)
val_ds = image_dataset_from_directory(
    val_dir,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True
)
test_ds = image_dataset_from_directory(
    test_dir,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False
)

# Extract class names
class_names = train_ds.class_names
print("Class names:", class_names)

# Build and compile the MobileNetV2 model
def build_mobilenetv2():
    base_model = keras.applications.MobileNetV2(
        weights='imagenet',
        input_shape=(img_height, img_width, 3),
        include_top=False
    )
    base_model.trainable = True
    for layer in base_model.layers[:-5]:  # Unfreeze only top layers
        layer.trainable = False

    inputs = keras.Input(shape=(img_height, img_width, 3))
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(len(class_names), activation='softmax')(x)

    model = keras.Model(inputs, x)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model

# Initialize the MobileNetV2 model
model_mobilenetv2 = build_mobilenetv2()

# Learning rate scheduler and early stopping callbacks
lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', mode="max", patience=5, restore_best_weights=True, verbose=1)

# Train the model
history_mobilenetv2 = model_mobilenetv2.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[lr_scheduler, early_stopping]
)

# Save the trained model
model_mobilenetv2.save("mobilenetv2.keras")

# Evaluate on test set
test_loss, test_accuracy = model_mobilenetv2.evaluate(test_ds)
print(f"MobileNetV2 Test Accuracy: {test_accuracy:.4f}")

# Generate predictions and evaluate
test_images, test_labels, image_paths = [], [], []

# Extract the image paths from the test dataset
image_paths = [str(path) for path in test_dir.glob('*/*.jpg')]

for image_batch, label_batch in test_ds:
    test_images.append(image_batch.numpy())
    test_labels.append(label_batch.numpy())

test_images = np.concatenate(test_images)
test_labels = np.concatenate(test_labels)

# Make predictions
mobilenetv2_predictions = model_mobilenetv2.predict(test_images)
predicted_labels_mobilenetv2 = np.argmax(mobilenetv2_predictions, axis=1)
prediction_probabilities = np.max(mobilenetv2_predictions, axis=1)

# Create a DataFrame to hold the results
results_df = pd.DataFrame({
    "Image Path": [pathlib.Path(image_path).name for image_path in image_paths],
    "True Label": [test_mapping[pathlib.Path(image_path).name] for image_path in image_paths],
    "Predicted Label": [class_names[pred] for pred in predicted_labels_mobilenetv2],
    "Prediction Probability": prediction_probabilities
})

# Save results to a CSV file
results_df.to_csv("mobilenetv2_results.csv", index=False)
print("Results saved to mobilenetv2_results.csv")

# Print the results DataFrame
print(results_df)

# Classification report
print("MobileNetV2 Classification Report:")
print(classification_report(test_labels, predicted_labels_mobilenetv2, target_names=class_names))

# Confusion matrix
conf_matrix_mobilenetv2 = confusion_matrix(test_labels, predicted_labels_mobilenetv2)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_mobilenetv2, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title("MobileNetV2 Confusion Matrix Heatmap")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
