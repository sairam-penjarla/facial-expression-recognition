from datasets import load_dataset
from code.utilities.common_utils import CommonUtils
from datasets import DatasetDict, ClassLabel
from datasets import load_dataset,load_metric,concatenate_datasets

from code.logging import logger  # Importing logger module for logging

class DataGathering:
    def __init__(self, config: dict) -> None:
        self.common_utils = CommonUtils()
        self.config = config

    def run(self):
        af_dataset = load_dataset("dataset", data_dir="training_data/affectnet_training_data")

        # Get the label names
        label_names = af_dataset['train'].features['label'].names

        # Get the index of the 'contempt' label
        contempt_index = label_names.index('contempt')

        # Filter out the rows with 'contempt' label
        filtered_dataset = af_dataset['train'].filter(lambda example: example['label'] != contempt_index)

        # Remove 'contempt' from the label names
        new_label_names = [name for name in label_names if name != 'contempt']

        # Create a new ClassLabel feature without 'contempt'
        new_label_feature = ClassLabel(names=new_label_names)

        # Function to remap labels to the new indices
        def remap_label(label):
            return new_label_feature.str2int(label_names[label])

        # Update the dataset's label feature
        filtered_dataset = filtered_dataset.map(lambda example: {'label': remap_label(example['label'])})

        # Update the dataset features
        filtered_dataset = filtered_dataset.cast_column('label', new_label_feature)

        # Update the DatasetDict
        af_dataset = DatasetDict({'train': filtered_dataset})

        # Get the label names
        label_names = af_dataset['train'].features['label'].names

        # Check if 'anger' exists in label_names before changing it
        if 'anger' in label_names:
            # Change 'anger' to 'angry' in label names
            new_label_names = [name if name != 'anger' else 'angry' for name in label_names]

            # Update the label names in the dataset
            af_dataset['train'] = af_dataset['train'].rename_column('label', 'old_label')
            af_dataset['train'] = af_dataset['train'].rename_column('old_label', 'label')

            # Update the label 'anger' to 'angry' in the dataset
            af_dataset['train'] = af_dataset['train'].map(lambda example: {'label': 'angry' if example['label'] == 'anger' else example['label']})

            # Update the ClassLabel feature
            new_label_feature = ClassLabel(names=new_label_names)

            # Update the dataset's label feature
            af_dataset['train'] = af_dataset['train'].cast_column('label', new_label_feature)

        mmi_dataset = load_dataset("imagefolder", data_dir="/training_data/mma-facial-expression")
        fer_dataset = load_dataset("imagefolder", data_dir="/training_data/fer2013")
        
        combined_dataset = concatenate_datasets([af_dataset['train'],fer_dataset['train'],mmi_dataset['train']])
        
        dataset = DatasetDict()
        dataset['train'] = combined_dataset
        return dataset