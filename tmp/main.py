import logging

import evaluate
from datasets import DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)

import load_dataset

logger = logging.getLogger(__name__)


def tokenize_dataset(
    tokenizer: PreTrainedTokenizer, dataset_dict: DatasetDict
) -> DatasetDict:
    def function(batch):
        return tokenizer(
            text=batch["text"], text_pair=batch["text_pair"], truncation=True
        )

    dataset_dict = dataset_dict.map(function=function, batched=True)
    column_names = ["attention_mask", "input_ids", "label", "token_type_ids"]
    return dataset_dict.select_columns(column_names)


def main():
    dataset = "scitail"
    pretrained_model_name_or_path = "bert-base-cased"
    output_dir = f"/tmp/{pretrained_model_name_or_path}-{dataset}"

    training_arguments = TrainingArguments(output_dir, optim="adamw_torch")

    dataset_dict = load_dataset.load_dataset(dataset)
    dataset_dict = dataset_dict.filter((lambda _, idx: idx == 0), with_indices=True)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    dataset_dict = tokenize_dataset(tokenizer, dataset_dict)

    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path
        )

    evaluation_module = evaluate.load("accuracy")

    def compute_metrics(eval_prediction: EvalPrediction):
        predictions, references = eval_prediction
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        return evaluation_module.compute(
            predictions=predictions.argmax(axis=1), references=references
        )

    # if "train" in dataset_dict and "validation" in dataset_dict:
    trainer = Trainer(
        args=training_arguments,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        tokenizer=tokenizer,
        model_init=model_init,
        compute_metrics=compute_metrics,
    )

    train_output = trainer.train()
    print(train_output)

    best_run = trainer.hyperparameter_search(direction="maximize")
    print(best_run)

    if "test" in dataset_dict:
        prediction_output = trainer.predict(dataset_dict["test"])
        print(prediction_output)


main()
