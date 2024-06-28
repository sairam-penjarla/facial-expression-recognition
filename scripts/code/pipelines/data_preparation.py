from code.logging import logger  # Importing logger module for logging
from code.utilities.common_utils import CommonUtils
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer, pipeline
from torchvision.transforms import (  # Import image transformation functions
    CenterCrop,  # Center crop an image
    Compose,  # Compose multiple image transformations
    Normalize,  # Normalize image pixel values
    RandomRotation,  # Apply random rotation to images
    RandomResizedCrop,  # Crop and resize images randomly
    RandomHorizontalFlip,  # Apply random horizontal flip
    RandomAdjustSharpness,  # Adjust sharpness randomly
    Resize,  # Resize images
    ToTensor  # Convert images to PyTorch tensors
)

class DataPreparation:
    def __init__(self, config: dict) -> None:
        self.common_utils = CommonUtils()
        self.config = config
        self.MODEL_NAME = config['params']['model_name']

    def run(self, dataset):
        labels = dataset["train"].features["label"].names
        label2id, id2label = dict(), dict()
        for i, label in enumerate(labels):
            label2id[label] = i
            id2label[i] = label
        
        image_processor  = AutoImageProcessor.from_pretrained(self.MODEL_NAME)
        # Retrieve the image mean and standard deviation used for normalization
        image_mean, image_std = image_processor.image_mean, image_processor.image_std

        # Get the size (height) of the ViT model's input images
        size = image_processor.size["height"]
        print("Size: ", size)

        # Define a normalization transformation for the input images
        normalize = Normalize(mean=image_mean, std=image_std)

        # Define a set of transformations for training data
        train_tf = Compose(
            [
                Resize((size, size)),             # Resize images to the ViT model's input size
                RandomRotation(90),               # Apply random rotation
                RandomAdjustSharpness(2),         # Adjust sharpness randomly
                RandomHorizontalFlip(0.5),        # Random horizontal flip
                ToTensor(),                       # Convert images to tensors
                normalize                          # Normalize images using mean and std
            ]
        )
        # Define a set of transformations for validation data
        val_tf = Compose(
            [
                Resize((size, size)),             # Resize images to the ViT model's input size
                ToTensor(),                       # Convert images to tensors
                normalize                         # Normalize images using mean and std
            ]
        )
        # Define a function to apply training transformations to a batch of examples
        def train_transforms(examples):
            examples['pixel_values'] = [train_tf(image.convert("RGB")) for image in examples['image']]
            return examples

        # Define a function to apply validation transformations to a batch of examples
        def val_transforms(examples):
            examples['pixel_values'] = [val_tf(image.convert("RGB")) for image in examples['image']]
            return examples
        
        splits = dataset["train"].train_test_split(test_size=0.2)
        train_data = splits['train']
        val_data = splits['test']

        train_data.set_transform(train_transforms)
        val_data.set_transform(val_transforms)

        return train_data, val_data, label2id, id2label