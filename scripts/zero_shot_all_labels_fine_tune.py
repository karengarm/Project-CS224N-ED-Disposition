from transformers import pipeline
import numpy as np
import pandas as pd
from sklearn import metrics
import torch
import datasets
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm

model_dir = "/root/models/bart_frozen_base_3_12_23"


# https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/pipelines#transformers.ZeroShotClassificationPipeline
pipe = pipeline(
    task="zero-shot-classification",
    model=model_dir,
    tokenizer="facebook/bart-large-mnli",
    framework="pt",
    device=0,
#     batch_size=10,
)
assert pipe.device.type == "cuda"
print(pipe)

from datasets import load_dataset
dataset = load_dataset("csv", data_files="/root/data/chexbert_results.csv")

# labels = ["Fracture", "Edema", "Cardiomegaly", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion"]
labels = [
	"Pneumonia", # XR CHEST and CT CHEST
	"Pneumothorax", # XR CHEST and CT CHEST
	"Pleural Effusion", # XR CHEST and CT CHEST
	"Edema", # "Pulmonary edema", "Cerebral edema"; XR CHEST and CT CHEST
	"Fracture", # "Rib fracture", "Skull fracture"; XR CHEST and CT CHEST
	"Infection",
	"Aspiration",
	"Cardiomegaly",
	"Opacities",
	"Atelectasis",
	"Intracranial hemorrhage",
	"Subarachnoid hemorrhage",
	"Subdural hemorrhage",
	"Epidural hemorrhage",
	"Intraparenchymal hemorrhage",
	"Intraventricular hemorrhage",
	"Stroke",
	"Diffuse axonal injury",
	"Appendicitis ",
	"Cholecystitis",
	"Abdominal Aortic Aneurysm",
	"Small bowel obstruction",
	"Pancreatitis",
	"Splenic laceration",
	"Liver laceration",
	"Colitis",
	"Pyelonephritis",
	"Nephrolithiasis",
	"Malignancy",
	"Pericaridial effusion",
	"Aortic dissection",
]
print("num labels:", len(labels))

# key_dataset = KeyDataset(dataset["train"].select(range(10)), "Report Impression")
key_dataset = KeyDataset(dataset["train"], "Report Impression")

pred_all = pipe(
    sequences=key_dataset,
    candidate_labels=labels,
    multi_label=True,
    batch_size=16,
)

results = []
partial_results = []
for i, out in enumerate(tqdm(pred_all, total=len(key_dataset))):
    results.append(out)
    partial_results.append(out)
    
    if i % 1000 == 0:
        partial_results = Dataset.from_list(partial_results)
        partial_results.save_to_disk(f"/root/data/zero_shot_predictions_fine_tune/checkpoint_{i}")
        partial_results = []
        print(f"saved partial results {i}")
    
print(len(results))

predictions = Dataset.from_list(results)
predictions.save_to_disk("/root/data/zero_shot_predictions_fine_tune/all")