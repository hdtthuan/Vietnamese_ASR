import os
import sys
import random
import warnings
from dataclasses import dataclass
from typing import List, Dict, Union

import numpy as np
import librosa
import torch
from datasets import load_dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    TrainerCallback,
)
from transformers.utils import logging as hf_logging

# Suppress warnings
warnings.filterwarnings('ignore')
hf_logging.set_verbosity_error()
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

# WER metric
try:
    import evaluate
    wer_metric = evaluate.load("wer")
    _HAS_EVAL = True
except Exception:
    wer_metric = None
    _HAS_EVAL = False

# ========== CONFIG ==========
BASE_DIR = "/kaggle/input/dsp-vimd/ViMD_Dataset_Processed"
TRAIN_CSV = os.path.join(BASE_DIR, "train", "metadata.csv")
VALID_CSV = os.path.join(BASE_DIR, "valid", "metadata.csv")
TEST_CSV  = os.path.join(BASE_DIR, "test",  "metadata.csv")

# Sử dụng model từ VinAI
MODEL_ID = "vinai/PhoWhisper-base"
OUTPUT_DIR = "/kaggle/working/phowhisper-finetuned"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== HYPERPARAMETERS ==========
LEARNING_RATE = 2e-6
TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
EVAL_BATCH_SIZE = 8
SEED = 42
WARMUP_RATIO = 0.1
NUM_TRAIN_EPOCHS = 3
FP16 = True
SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH = 30.0
MAX_EVAL_SAMPLES = 300

# Audio augmentation
USE_AUDIO_AUGMENTATION = False

# ========== WandB Setup ==========
try:
    _REPORT_TO = "wandb"
    WANDB_RUN_NAME = f"phowhisper-accent-{random.randint(1000,9999)}"
    WANDB_PROJECT = "phowhisper-vietnamese-accents"

    os.environ["WANDB_API_KEY"] = "e896b413a2e8bfb2509f89a27cf130ab7dd54840"
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    os.environ["WANDB_RUN_NAME"] = WANDB_RUN_NAME
    os.environ["WANDB_LOG_MODEL"] = "end"
    os.environ["WANDB_START_METHOD"] = "thread"
    os.environ["WANDB_SILENT"] = "true"

    print("WandB configured for server environment")
except Exception as e:
    print("⚠️ WandB not configured, running without logging")

set_seed(SEED)

# ========== LOAD DATA ==========
print("\n" + "="*60)
print("LOADING DATASET")
print("="*60)

data_files = {"train": TRAIN_CSV, "validation": VALID_CSV, "test": TEST_CSV}
ds = load_dataset("csv", data_files=data_files)

print(f"✅ Dataset loaded:")
print(f"  • Train:      {len(ds['train']):>6} samples")
print(f"  • Validation: {len(ds['validation']):>6} samples")
print(f"  • Test:       {len(ds['test']):>6} samples")

# ========== LOAD MODEL & PROCESSOR ==========
print("\n" + "="*60)
print("LOADING MODEL & PROCESSOR")
print("="*60)

processor = WhisperProcessor.from_pretrained(MODEL_ID, language="vi", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)

# Configure for Vietnamese
model.generation_config.language = "vi"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None

print(f"Model: {MODEL_ID}")
print(f"Language: Vietnamese (vi)")
print(f"Task: Transcribe")

# ========== PATH NORMALIZATION ==========
def normalize_path(p: str) -> str:
    if not isinstance(p, str): 
        return p
    if p.startswith("/kaggle/input/data-vimd/ViMD_Dataset_Processed"):
        tail = p.split("ViMD_Dataset_Processed", 1)[-1].lstrip("/")
        return os.path.join(BASE_DIR, tail)
    if not os.path.isabs(p):
        return os.path.join(BASE_DIR, p)
    return p

# ========== AUDIO AUGMENTATION ==========
def augment_audio(audio_array, sr):
    """Audio augmentation for accent robustness"""
    if not USE_AUDIO_AUGMENTATION or random.random() > 0.3:
        return audio_array
    
    aug_type = random.choice(['pitch', 'speed', 'noise'])
    
    try:
        if aug_type == 'pitch':
            n_steps = random.uniform(-2, 2)
            audio_array = librosa.effects.pitch_shift(audio_array, sr=sr, n_steps=n_steps)
        elif aug_type == 'speed':
            rate = random.uniform(0.9, 1.1)
            audio_array = librosa.effects.time_stretch(audio_array, rate=rate)
        elif aug_type == 'noise':
            noise = np.random.randn(len(audio_array)) * 0.005
            audio_array = audio_array + noise
    except:
        pass
    
    return audio_array

# ========== PREPARE DATASET ==========
def prepare_dataset(batch):
    input_features_list = []
    labels_list = []
    skipped = 0
    
    audio_paths = batch.get("audio_path") or batch.get("audio") or batch.get("path")
    texts = batch["text"]
    
    for audio_p, txt in zip(audio_paths, texts):
        p = normalize_path(audio_p)
        
        if not os.path.isfile(p) or not txt or not isinstance(txt, str) or len(txt.strip()) == 0:
            skipped += 1
            continue
        
        try:
            audio_array, sr = librosa.load(p, sr=SAMPLE_RATE, mono=True)
            
            if len(audio_array) / SAMPLE_RATE > MAX_AUDIO_LENGTH:
                skipped += 1
                continue
            
            audio_array = augment_audio(audio_array, SAMPLE_RATE)
            
            input_features = processor.feature_extractor(
                audio_array, 
                sampling_rate=SAMPLE_RATE,
                return_tensors="np"
            ).input_features[0]
            
            labels = processor.tokenizer(
                txt.strip(),
                truncation=True,
                max_length=448,
                padding=False,
                add_special_tokens=True
            ).input_ids
            
            if len(labels) < 2:
                skipped += 1
                continue
                
            input_features_list.append(input_features)
            labels_list.append(labels)
            
        except Exception:
            skipped += 1
            continue
    
    return {
        "input_features": input_features_list,
        "labels": labels_list
    }

# ========== PROCESS DATASETS ==========
print("\n" + "="*60)
print("PROCESSING DATASETS")
print("="*60)

print("Processing train dataset...")
ds["train"] = ds["train"].map(
    prepare_dataset,
    batched=True,
    batch_size=16,
    remove_columns=ds["train"].column_names,
    num_proc=2,
    desc="Train"
)

print("Processing validation dataset...")
ds["validation"] = ds["validation"].map(
    prepare_dataset,
    batched=True,
    batch_size=16,
    remove_columns=ds["validation"].column_names,
    num_proc=2,
    desc="Validation"
)

if len(ds["validation"]) > MAX_EVAL_SAMPLES:
    print(f"Subsampling validation: {len(ds['validation'])} → {MAX_EVAL_SAMPLES}")
    ds["validation"] = ds["validation"].shuffle(seed=SEED).select(range(MAX_EVAL_SAMPLES))

print(f"\nProcessing complete:")
print(f"  • Train samples:      {len(ds['train']):>6}")
print(f"  • Validation samples: {len(ds['validation']):>6}")

# ========== DATA COLLATOR ==========
@dataclass
class WhisperDataCollator:
    processor: WhisperProcessor
    
    def __call__(self, features: List[Dict[str, Union[List[int], np.ndarray]]]):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        
        batch["labels"] = labels
        return batch

collator = WhisperDataCollator(processor=processor)

# ==========================
# COLLATOR & METRICS
# ==========================
@dataclass
class SpeechDataCollator:
    processor: AutoProcessor
    def __call__(self, features: List[Dict[str, Union[List[int], np.ndarray]]]):
        input_feats = [f["input_features"] for f in features]
        batch = self.processor.feature_extractor.pad({"input_features": input_feats}, return_tensors="pt")
        if batch["input_features"].ndim == 4:
            batch["input_features"] = batch["input_features"].squeeze(1)
        labels = [f["labels"] for f in features]
        label_batch = self.processor.tokenizer.pad({"input_ids": labels}, return_tensors="pt")
        batch["labels"] = label_batch["input_ids"].masked_fill(label_batch["attention_mask"] != 1, -100)
        return batch

collator = SpeechDataCollator(processor)

if _HAS_EVAL and wer_metric is not None:
    def compute_metrics_fn(pred):
        preds = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        decoded_preds = processor.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(pred.label_ids != -100, pred.label_ids, processor.tokenizer.pad_token_id)
        decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
        wer = wer_metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {"wer": wer}
else:
    def compute_metrics_fn(pred):
        return {}

wer_metric = evaluate.load("wer")

def compute_metrics_fn(pred):
    preds = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    decoded_preds = processor.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(pred.label_ids != -100, pred.label_ids, processor.tokenizer.pad_token_id)
    decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
    return {"wer": wer_metric.compute(predictions=decoded_preds, references=decoded_labels)}

# ==========================
# TRAINER CONFIG
# ==========================
args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    lr_scheduler_type="linear",
    warmup_ratio=WARMUP_RATIO,
    fp16=FP16,
    evaluation_strategy="steps",
    eval_steps=500,
    logging_steps=50,
    save_steps=500,
    save_total_limit=3,
    report_to=_REPORT_TO,
    run_name=os.environ.get("WANDB_RUN_NAME", "phowhisper-train"),
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    predict_with_generate=True,
    seed=SEED,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    data_collator=collator,
    compute_metrics=compute_metrics_fn,
    tokenizer=processor.feature_extractor,
)

print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)

trainer.train()
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

print(f"Model and processor saved to: {OUTPUT_DIR}")