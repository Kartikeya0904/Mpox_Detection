import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib

# Function to load disease mappings from a CSV file
def load_disease_mapping(file_path):
    mapping_df = pd.read_csv(file_path)
    return dict(zip(mapping_df['Filename'], mapping_df['Disease']))

# Define dataset paths
train_dir = pathlib.Path("C:/MPOX CODE/DATA_AFTER_SPLIT/aug_train")
val_dir = pathlib.Path("C:/MPOX CODE/DATA_AFTER_SPLIT/Valid")
test_dir = pathlib.Path("C:/MPOX CODE/DATA_AFTER_SPLIT/Test")


# Load disease mappings
train_mapping = load_disease_mapping("C:/MPOX CODE/train_disease_mapping.csv")
val_mapping = load_disease_mapping("C:/MPOX CODE/val_mapping.csv")
test_mapping = load_disease_mapping("C:/MPOX CODE/test_mapping.csv")
# Set parameters
batch_size = 32
img_height = 224
img_width = 224
epochs = 15
initial_lr = 1e-4

# Load datasets
train_ds = image_dataset_from_directory(
    train_dir,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
val_ds = image_dataset_from_directory(
    val_dir,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
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

# Model building function for EfficientNetB0
def build_model():
    base_model = keras.applications.EfficientNetB0(
        weights='imagenet',
        input_shape=(img_height, img_width, 3),
        include_top=False
    )
    
    # Freeze all layers in the base model
    base_model.trainable = False
    
    inputs = keras.Input(shape=(img_height, img_width, 3))
    x = base_model(inputs, training=False)  # Set training=False to prevent BatchNorm updates
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(len(class_names), activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=initial_lr),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model

# Initialize the EfficientNetB0 model
model_efficientnet = build_model()

# Callbacks for learning rate scheduling and early stopping
callbacks = [
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
]

# Train EfficientNetB0
print("Training EfficientNetB0...")
history_efficientnet = model_efficientnet.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)


model_efficientnet.save("C:/MPOX CODE/efficientnet.keras")

# Evaluate EfficientNetB0 on test set
test_loss, test_accuracy = model_efficientnet.evaluate(test_ds)
print(f"EfficientNetB0 Test Accuracy: {test_accuracy:.4f}")

# Function to evaluate and save results
def evaluate_and_save_results(model, test_ds, class_names, test_mapping, output_file):
    test_images, test_labels, image_paths = [], [], []
    for images, labels in test_ds:
        test_images.append(images.numpy())
        test_labels.append(labels.numpy())
    
    test_images = np.concatenate(test_images)
    test_labels = np.concatenate(test_labels)
    
    # Predictions
    predictions = model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)
    prediction_probabilities = np.max(predictions, axis=1)

    # DataFrame with results
    results_df = pd.DataFrame({
        "Image Path": [pathlib.Path(path).name for path in test_ds.file_paths],
        "True Label": [test_mapping[pathlib.Path(path).name] for path in test_ds.file_paths],
        "Predicted Label": [class_names[pred] for pred in predicted_labels],
        "Prediction Probability": prediction_probabilities
    })
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    # Confusion Matrix
    conf_matrix = confusion_matrix(test_labels, predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix for EfficientNetB0")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    # Classification report
    print("Classification Report:")
    print(classification_report(test_labels, predicted_labels, target_names=class_names))

# Evaluate and save results for EfficientNetB0
evaluate_and_save_results(model_efficientnet, test_ds, class_names, test_mapping, "C:/MPOXCLASS/efficientnet_results.csv")
