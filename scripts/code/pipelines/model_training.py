from code.logging import logger  # Importing logger module for logging
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer, pipeline
from code.utilities.common_utils import CommonUtils

class ModelTraining():
    def __init__(self, config: dict) -> None:
        self.common_utils = CommonUtils()
        self.config = config
        self.MODEL_NAME = config['params']['model_name']
    
    def run(self, train_data, val_data, label2id, id2label) -> None:
        model = AutoModelForImageClassification.from_pretrained(
            self.MODEL_NAME, 
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes = True, # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
        )
        model_name = self.MODEL_NAME.split("/")[-1]
        args = TrainingArguments(
                f"{model_name}",
                remove_unused_columns=False,
                evaluation_strategy="steps",
                save_strategy="steps",  # Align with evaluation_strategy
                learning_rate=3e-5,
                lr_scheduler_type="cosine",
                auto_find_batch_size=True,
                per_device_train_batch_size=32,
                gradient_accumulation_steps=8,
                per_device_eval_batch_size=32,
                weight_decay=0.1,
                num_train_epochs=3,
                warmup_steps=1000,
                logging_steps=50,
                eval_steps=100,
                load_best_model_at_end=True,
                metric_for_best_model="accuracy",
                push_to_hub=True,
                report_to="none"
            )
        image_processor  = AutoImageProcessor.from_pretrained(self.MODEL_NAME)
        trainer = Trainer(
            model,
            args,
            train_dataset=train_data,
            eval_dataset=val_data,
            tokenizer=image_processor,
            compute_metrics=self.common_utils.compute_metrics,
            data_collator=self.common_utils.collate_fn,
        )
        train_results = trainer.train()
        metrics = trainer.evaluate()
        # some nice to haves:
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        outputs = trainer.predict(val_data)
        print(outputs.metrics)