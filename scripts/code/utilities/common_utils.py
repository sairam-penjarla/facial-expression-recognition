from code.logging import logger  # Importing logger module for logging
from datasets import load_metric
import numpy as np
import torch

class CommonUtils:
    def __init__(self) -> None:
        self.metric = load_metric("accuracy")

    def compute_metrics(self, eval_pred):
        """Computes accuracy on a batch of predictions"""
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return self.metric.compute(predictions=predictions, references=eval_pred.label_ids)
    
    def collate_fn(self, examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}