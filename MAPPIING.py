import pathlib
import pandas as pd

# Define dataset paths
train_dir = pathlib.Path("C:/MPOX CODE/DATA_AFTER_SPLIT/aug_train")
val_dir = pathlib.Path("C:/MPOX CODE/DATA_AFTER_SPLIT/Valid")
test_dir = pathlib.Path("C:/MPOX CODE/DATA_AFTER_SPLIT/Test")

def map_images_to_disease(directory):
    image_label_mapping = []
    
    # Iterate through each subdirectory (disease)
    for label in directory.iterdir():
        if label.is_dir():  # Ensure it's a directory
            # Iterate through each image file in the subdirectory
            for image_file in label.glob('*'):
                if image_file.is_file():  # Ensure it's a file
                    # Append a tuple (filename, disease) to the list
                    image_label_mapping.append({
                        "Filename": image_file.name,
                        "Disease": label.name
                    })
    
    return image_label_mapping

# Map training, validation, and testing datasets
train_mapping = map_images_to_disease(train_dir)
val_mapping = map_images_to_disease(val_dir)
test_mapping = map_images_to_disease(test_dir)

# Convert mappings to DataFrames
train_df = pd.DataFrame(train_mapping)
val_df = pd.DataFrame(val_mapping)
test_df = pd.DataFrame(test_mapping)

# Save DataFrames to CSV files
train_df.to_csv("train_disease_mapping.csv", index=False)
val_df.to_csv("val_mapping.csv", index=False)
test_df.to_csv("test_mapping.csv", index=False)

print("Mappings saved as CSV files:")
print("- train_mapping.csv")
print("- val_mapping.csv")
print("- test_mapping.csv")
