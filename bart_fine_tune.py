model_dir = "/root/models/bart_all_3_11_23"


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


dataset = datasets.load_from_disk("/root/data/bart_fine_tune")
print(dataset)


small_train_dataset = dataset['train'].shuffle(seed=42)
small_val_dataset = dataset['val'].shuffle(seed=42).select(range(1000))



from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits_tuple, labels = eval_pred
    logits, _ = logits_tuple
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)



from transformers import TrainingArguments, Trainer

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