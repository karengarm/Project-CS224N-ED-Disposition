"""
THIS CODE WILL OOM!
Do not run this. Use scripts/zero_shot_all_labels_fine_tune.py instead!
"""


model_dir = "/root/models/bart_frozen_base_3_12_23"

from pynvml import *

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()
    
print_gpu_utilization()

from transformers import pipeline
import numpy as np
import pandas as pd
from sklearn import metrics
import torch
import datasets
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm
import os
import pickle
os.environ['DISABLE_MLFLOW_INTEGRATION'] = 'TRUE'

print_gpu_utilization()

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
print_gpu_utilization()

train_tokenized = datasets.load_from_disk("/root/data/bart_train_labels")
val_tokenized = datasets.load_from_disk("/root/data/bart_val_labels")
test_tokenized = datasets.load_from_disk("/root/data/bart_test_labels")


small_val_dataset = val_tokenized["val"].shuffle(seed=42).select(range(10))


from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits_tuple, labels = eval_pred
    logits, _ = logits_tuple
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    output_dir=f"{model_dir}/output",
    evaluation_strategy="epoch",
    num_train_epochs=1,
    logging_steps=10,
    per_device_train_batch_size=3,
    per_device_eval_batch_size=1, # TODO: adjust
    gradient_accumulation_steps=10, # effective batch size is per_device_train_batch_size * gradient_accumulation_steps
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=small_val_dataset,
    compute_metrics=compute_metrics,
)

# for dataset, name in zip([small_val_dataset], ["zzz"]):
for dataset, name in zip([val_tokenized, test_tokenized, train_tokenized], ["val", "test", "train"]):
    for split in ["train", "val", "test"]:
        print(f"begin {name} {split}")
        predictions = trainer.predict(dataset[split])

        # open a file, where you want to store the data
        file = open(f'/root/data/bart_fine_tune_pred/{name}_split_{split}.pickle', 'wb')

        # dump information to that file
        pickle.dump(predictions, file)

        # close the file
        file.close()
        print(f"end {name} {split}")