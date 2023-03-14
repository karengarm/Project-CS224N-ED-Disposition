"""
Batch size can be increased, because we are freezing the base model and only tuning classification head
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

import os
os.environ['DISABLE_MLFLOW_INTEGRATION'] = 'TRUE'


from transformers import pipeline
import numpy as np
import pandas as pd
from sklearn import metrics
import torch
import datasets
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm


from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")

# freeze model.base_model but leave model.classification_head unfrozen
for param in model.base_model.parameters():
    param.requires_grad = False

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("trainable params:", trainable_params, trainable_params / total_params)

train_tokenized = datasets.load_from_disk("/root/data/bart_fine_tune_train_labels")
val_tokenized = datasets.load_from_disk("/root/data/bart_fine_tune_val_labels")

# print(train_tokenized)
# print(val_tokenized)

from datasets import concatenate_datasets
train_concat = concatenate_datasets([train_tokenized['train'], train_tokenized['val']])
val_concat = concatenate_datasets([val_tokenized['train'], val_tokenized['val']])


small_train_dataset = train_concat.shuffle(seed=42)
small_val_dataset = val_concat.shuffle(seed=42).select(range(1000))



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
    per_device_eval_batch_size=3,
    gradient_accumulation_steps=10, # effective batch size is per_device_train_batch_size * gradient_accumulation_steps
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_val_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()


model.save_pretrained(model_dir)


import json
json_object = json.dumps(trainer.state.log_history, indent=4)

with open(f"{model_dir}/log_history.json", "w") as outfile:
    outfile.write(json_object)