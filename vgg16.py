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
# Define dataset paths
train_dir = pathlib.Path("C:/MPOX CODE/DATA_AFTER_SPLIT/aug_train")
val_dir = pathlib.Path("C:/MPOX CODE/DATA_AFTER_SPLIT/Valid")
test_dir = pathlib.Path("C:/MPOX CODE/DATA_AFTER_SPLIT/Test")

# Load disease mappings from CSV files
def load_disease_mapping(file_path):
    mapping_df = pd.read_csv(file_path)
    return dict(zip(mapping_df['Filename'], mapping_df['Disease']))

# Load mappings for train, val, and test datasets
train_mapping = load_disease_mapping("C:/MPOXCLASS/train_disease_mapping.csv")
val_mapping = load_disease_mapping("C:/MPOXCLASS/val_mapping.csv")
test_mapping = load_disease_mapping("C:/MPOXCLASS/test_mapping.csv")

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

# Build and compile the VGG16 model
def build_vgg16():
    base_model = keras.applications.VGG16(
        weights='imagenet',
        input_shape=(img_height, img_width, 3),
        include_top=False
    )
    base_model.trainable = True
    for layer in base_model.layers[:-5]:  # Unfreeze only top layers
        layer.trainable = False

    inputs = keras.Input(shape=(img_height, img_width, 3))
    x = keras.applications.vgg16.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(len(class_names), activation='softmax')(x)

    model = keras.Model(inputs, x)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model

# Initialize the VGG16 model
model_vgg16 = build_vgg16()

# Learning rate scheduler and early stopping callbacks
lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy',mode = "max" ,patience=5, restore_best_weights=True, verbose=1)

# Train the model
history_vgg16 = model_vgg16.fit(
    train_ds, 
    validation_data=val_ds, 
    epochs=epochs,
    callbacks=[lr_scheduler, early_stopping]
)

# Save the trained model
model_vgg16.save("vgg16.keras")

# Evaluate on test set
test_loss, test_accuracy = model_vgg16.evaluate(test_ds)
print(f"VGG16 Test Accuracy: {test_accuracy:.4f}")

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
vgg16_predictions = model_vgg16.predict(test_images)
predicted_labels_vgg16 = np.argmax(vgg16_predictions, axis=1)
prediction_probabilities = np.max(vgg16_predictions, axis=1)

# Create a DataFrame to hold the results
results_df = pd.DataFrame({
    "Image Path": [pathlib.Path(image_path).name for image_path in image_paths],  # Only get the filename
    "True Label": [test_mapping[pathlib.Path(image_path).name] for image_path in image_paths],
    "Predicted Label": [class_names[pred] for pred in predicted_labels_vgg16],
    "Prediction Probability": prediction_probabilities
})

# Save results to a CSV file
results_df.to_csv("vgg16_results.csv", index=False)
print("Results saved to vgg16_results.csv")

# Print the results DataFrame
print(results_df)

# Classification report
print("VGG16 Classification Report:")
print(classification_report(test_labels, predicted_labels_vgg16, target_names=class_names))

# Confusion matrix
conf_matrix_vgg16 = confusion_matrix(test_labels, predicted_labels_vgg16)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_vgg16, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title("VGG16 Confusion Matrix Heatmap")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
