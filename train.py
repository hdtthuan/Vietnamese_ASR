import os, glob, re
import sys
import random
import warnings
import librosa, soundfile as sf
from dataclasses import dataclass
from typing import List, Dict, Union, Any
import evaluate
import pandas as pd
from tqdm import tqdm

import gc
import numpy as np
import librosa
import torch
from datasets import load_dataset, Dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    TrainerCallback,
)
from transformers.utils import logging as hf_logging

from kaggle_secrets import UserSecretsClient
_REPORT_TO = "none"
WANDB_RUN_NAME = f"phowhisper-accent-{random.randint(1000,9999)}"
WANDB_PROJECT = "phowhisper-vietnamese-accents"
try:
    user_secrets = UserSecretsClient()
    key = user_secrets.get_secret("wandb")
    os.environ["WANDB_API_KEY"] = key
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    os.environ["WANDB_RUN_NAME"] = WANDB_RUN_NAME
    os.environ["WANDB_LOG_MODEL"] = "end"
    os.environ["WANDB_START_METHOD"] = "thread"
    os.environ["WANDB_SILENT"] = "true"
    _REPORT_TO = "wandb"
    print("✅ WandB configured")
except Exception as e:
    print("⚠️ WandB not configured, running without logging")

MODEL_ID = "vinai/PhoWhisper-base"
BASE_DATA_DIR = "/kaggle/input/vimddata/ViMD_Dataset"
OUTPUT_DIR = "/kaggle/working/train_outputs/phowhisper_vimd"
os.makedirs(OUTPUT_DIR, exist_ok=True)

target_sr = 16000
min_duration = 1.0
max_duration = 30.0
trim_top_db = 25
normalize_peak = 0.99

LEARNING_RATE = 2e-6
TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
EVAL_BATCH_SIZE = 4
SEED = 42
WARMUP_RATIO = 0.1
NUM_TRAIN_EPOCHS = 10
FP16 = True
SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH = 30.0
MAX_EVAL_SAMPLES = 200

# Audio augmentation toggle
USE_AUDIO_AUGMENTATION = False

set_seed(SEED)

def normalize_text_light(text):
    if text is None:
        return ""
    t = str(text).strip()
    t = re.sub(r'\s+', ' ', t)
    t = re.sub(r'[\x00-\x1f\x7f]', '', t)
    t = t.lower()
    return t

# PREPROCESS DATA
splits = ["train", "valid", "test"]
processed_root = os.path.join(OUTPUT_DIR, "processed")
os.makedirs(processed_root, exist_ok=True)

for split in splits:
    folder = os.path.join(BASE_DATA_DIR, split)
    print(f"Processing split: {split}")

    arrow_files = sorted(glob.glob(os.path.join(folder, "*.arrow")))
    if not arrow_files:
        print(f"No .arrow files found in {folder}")
        continue

    out_audio_dir = os.path.join(processed_root, split, "audio")
    os.makedirs(out_audio_dir, exist_ok=True)
    metadata_csv = os.path.join(processed_root, split, "metadata.csv")

    rows = []
    for p in arrow_files:
        ds = Dataset.from_file(p).cast_column("audio", Audio())
        for i in tqdm(range(len(ds)), desc=os.path.basename(p)):
            try:
                sample = ds[i]
                audio = sample["audio"]
                data = audio["array"]
                sr = audio["sampling_rate"]

                if data is None or len(data) == 0:
                    continue

                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                if sr != target_sr:
                    data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
                    sr = target_sr

                try:
                    data, _ = librosa.effects.trim(data, top_db=trim_top_db)
                except Exception:
                    pass

                duration = len(data) / sr
                if duration < min_duration or duration > max_duration:
                    continue

                peak = np.max(np.abs(data)) if data.size > 0 else 0.0
                if peak > 0:
                    data = (data / peak) * normalize_peak

                filename = sample.get("filename") or f"{split}_{os.path.basename(p)}_{i}.wav"
                filename = "".join(c for c in filename if c.isalnum() or c in ("_", "-", "."))
                out_path = os.path.join(out_audio_dir, filename)
                sf.write(out_path, data, sr, subtype="PCM_16")

                rows.append({
                    "audio_path": out_path,
                    "text": normalize_text_light(sample.get("text", "")),
                    "region": sample.get("region"),
                    "province_name": sample.get("province_name"),
                    "speakerID": sample.get("speakerID"),
                    "gender": sample.get("gender"),
                    "duration": duration,
                })
            except Exception as e:
                print(f"Error in {i}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(metadata_csv, index=False)
    print(f"✅ Saved metadata for {split} → {metadata_csv} ({len(df)} samples)")

# LOAD DATASETS
data_files = {
    "train": os.path.join(processed_root, "train", "metadata.csv"),
    "validation": os.path.join(processed_root, "valid", "metadata.csv"),
    "test": os.path.join(processed_root, "test", "metadata.csv"),
}
ds = load_dataset("csv", data_files=data_files)

# LOAD MODEL & PROCESSOR
processor = WhisperProcessor.from_pretrained(MODEL_ID, language="vi", task="transcribe")

# Load model in float32 first, then convert if needed
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_ID,
    low_cpu_mem_usage=True,
)

# Configure for Vietnamese
model.generation_config.language = "vi"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None

print(f"Model: {MODEL_ID}")
print(f"Language: Vietnamese (vi)")
print(f"Task: Transcribe")
print(f"FP16 will be handled by Trainer")

if processor.tokenizer.pad_token_id is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

# PATH NORMALIZATION
def normalize_path(p: str) -> str:
    if not isinstance(p, str): 
        return p
    if p.startswith("/kaggle/input/data-vimd/ViMD_Dataset_Processed"):
        tail = p.split("ViMD_Dataset_Processed", 1)[-1].lstrip("/")
        return os.path.join(BASE_DATA_DIR, tail)
    if not os.path.isabs(p):
        return os.path.join(BASE_DATA_DIR, p)
    return p

# AUDIO AUGMENTATION
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

# PREPARE DATASET
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
    
# PROCESS DATASETS
print("PROCESSING DATASETS")

print("Processing train dataset...")
ds["train"] = ds["train"].map(
    prepare_dataset,
    batched=True,
    batch_size=16,
    remove_columns=ds["train"].column_names,
    num_proc=1,
    desc="Train"
)

print("Processing validation dataset...")
ds["validation"] = ds["validation"].map(
    prepare_dataset,
    batched=True,
    batch_size=16,
    remove_columns=ds["validation"].column_names,
    num_proc=1,
    desc="Validation"
)

# Subsample validation
if len(ds["validation"]) > MAX_EVAL_SAMPLES:
    print(f"Subsampling validation: {len(ds['validation'])} → {MAX_EVAL_SAMPLES}")
    ds["validation"] = ds["validation"].shuffle(seed=SEED).select(range(MAX_EVAL_SAMPLES))

print(f"Processing complete:")
print(f"Train samples:      {len(ds['train']):>6}")
print(f"Validation samples: {len(ds['validation']):>6}")

# DATA COLLATOR
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

# VIETNAMESE TEXT NORMALIZATION
import unicodedata
import re

def normalize_vietnamese_text(text):
    """
    Normalize Vietnamese text for WER/CER computation
    - Normalize unicode (NFC)
    - Lowercase
    - Remove punctuation
    - Remove extra spaces
    - Keep only Vietnamese characters and spaces
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Unicode normalization
    text = unicodedata.normalize('NFC', text)
    
    # Lowercase
    text = text.lower()
    
    # Remove punctuation and special characters
    # Keep: a-z, à-ỹ, đ, spaces
    text = re.sub(r'[^\sa-zàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]', '', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

# METRICS
def compute_metrics(pred):
    pred_ids = pred.predictions
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]
    
    label_ids = pred.label_ids
    label_ids = np.where(label_ids != -100, label_ids, processor.tokenizer.pad_token_id)
    
    # Decode
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    
    # Normalize Vietnamese text
    pred_str = [normalize_vietnamese_text(p) for p in pred_str]
    label_str = [normalize_vietnamese_text(l) for l in label_str]
    
    # Filter empty
    valid_pairs = [(p, l) for p, l in zip(pred_str, label_str) if p and l]
    if not valid_pairs:
        return {"wer": 100.0, "cer": 100.0}
    
    pred_str_valid, label_str_valid = zip(*valid_pairs)
    
    # Compute WER using jiwer (more reliable)
    wer = 100.0
    try:
        # Try to use jiwer first
        try:
            from jiwer import wer as jiwer_wer
            wer_raw = jiwer_wer(list(label_str_valid), list(pred_str_valid))
            wer = round(100 * wer_raw, 2)
        except ImportError:
            # Fallback to evaluate library
            if _HAS_EVAL and wer_metric:
                wer_raw = wer_metric.compute(
                    predictions=list(pred_str_valid), 
                    references=list(label_str_valid)
                )
                if wer_raw <= 1.0:
                    wer = round(100 * wer_raw, 2)
                else:
                    wer = round(wer_raw, 2)
    except Exception as e:
        # Simple fallback: exact match percentage
        try:
            exact_matches = sum(1 for p, r in zip(pred_str_valid, label_str_valid) if p == r)
            wer = round(100 * (1 - exact_matches / len(pred_str_valid)), 2)
        except:
            wer = 100.0
    
    # Compute CER (Character Error Rate)
    def compute_cer(pred, ref):
        """Levenshtein distance at character level"""
        pred_chars = list(pred.replace(" ", ""))
        ref_chars = list(ref.replace(" ", ""))
        
        if not ref_chars:
            return 0.0
        
        # DP matrix
        d = [[0] * (len(ref_chars) + 1) for _ in range(len(pred_chars) + 1)]
        
        for i in range(len(pred_chars) + 1):
            d[i][0] = i
        for j in range(len(ref_chars) + 1):
            d[0][j] = j
        
        for i in range(1, len(pred_chars) + 1):
            for j in range(1, len(ref_chars) + 1):
                cost = 0 if pred_chars[i-1] == ref_chars[j-1] else 1
                d[i][j] = min(
                    d[i-1][j] + 1,      # deletion
                    d[i][j-1] + 1,      # insertion
                    d[i-1][j-1] + cost  # substitution
                )
        
        return d[len(pred_chars)][len(ref_chars)] / len(ref_chars)
    
    cer_total = sum(compute_cer(p, l) for p, l in valid_pairs) / len(valid_pairs)
    cer_pct = round(100 * cer_total, 2)
    
    # Sample logging (15% chance)
    if random.random() < 0.15:
        print(f"Sample:")
        print(f"Pred: {pred_str[0][:80]}...")
        print(f"True: {label_str[0][:80]}...")
    
    print(f"WER: {wer:>6.2f}% | CER: {cer_pct:>6.2f}%")
    
    return {"wer": wer, "cer": cer_pct}

# CALLBACK
class ProgressCallback(TrainerCallback):
    def __init__(self):
        self.start_time = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        import time
        self.start_time = time.time()
        print("TRAINING STARTED")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
            
        step = state.global_step
        
        # Training loss
        if 'loss' in logs and isinstance(logs['loss'], (int, float)):
            if step % 100 == 0:
                print(f"  Step {step:>4}: Loss = {logs['loss']:.4f}")
        
        # Evaluation metrics
        if 'eval_wer' in logs:
            wer = logs['eval_wer']
            cer = logs.get('eval_cer', 'N/A')
            print(f"Evaluation at step {step}:")
            print(f"WER: {wer}%")
            print(f"CER: {cer}%")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        import time
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        epoch = int(state.epoch)
        print(f"Epoch {epoch} completed | Time: {hours}h {minutes}m")
    
    def on_train_end(self, args, state, control, **kwargs):
        import time
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        print(f"Total training time: {hours}h {minutes}m")

# TRAINING ARGS
total_steps = (len(ds["train"]) // (TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)) * NUM_TRAIN_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)

print("TRAINING CONFIGURATION")
print(f"Learning rate:          {LEARNING_RATE}")
print(f"Batch size per device:  {TRAIN_BATCH_SIZE}")
print(f"Gradient accumulation:  {GRADIENT_ACCUMULATION_STEPS}")
print(f"Effective batch size:   {TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"Number of epochs:       {NUM_TRAIN_EPOCHS}")
print(f"Total training steps:   ~{total_steps}")
print(f"Warmup steps:           {warmup_steps}")
print(f"FP16 training:          {FP16}")
print(f"Eval steps:             500")
print(f"Save steps:             500")

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    warmup_steps=warmup_steps,
    gradient_checkpointing=True,
    fp16=FP16,
    
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    logging_steps=50,
    save_total_limit=2,
    
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    
    predict_with_generate=True,
    generation_max_length=200,
    generation_num_beams=1,
    
    optim="adamw_torch",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
    
    report_to=_REPORT_TO,
    run_name=WANDB_RUN_NAME,
    
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
    eval_accumulation_steps=8,
    remove_unused_columns=False,
    label_names=["labels"],
    
    seed=SEED,
    logging_nan_inf_filter=False,
    disable_tqdm=False,
    
    # Memory optimization
    gradient_checkpointing_kwargs={"use_reentrant": False},
    bf16=False,  # Disable bf16, use fp16 only
)

# TRAINER
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    data_collator=collator,
    processing_class=processor.feature_extractor,
    compute_metrics=compute_metrics,
    callbacks=[ProgressCallback()],
)

# TRAIN
if __name__ == "__main__":
    # Train
    train_result = trainer.train()
    # Save model
    print("SAVING MODEL")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    print(f"Model saved to: {OUTPUT_DIR}")
    
    # Final evaluation
    print("FINAL EVALUATION")
    
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    
    print(f"Validation Results:")
    print(f"WER: {eval_metrics.get('eval_wer', 'N/A')}%")
    print(f"CER: {eval_metrics.get('eval_cer', 'N/A')}%")
    
    # Test evaluation
    print("Processing test set...")
    ds["test"] = ds["test"].map(
        prepare_dataset,
        batched=True,
        batch_size=16,
        remove_columns=ds["test"].column_names,
        num_proc=1,
        desc="Test"
    )
    
    if len(ds["test"]) > 0:
        test_metrics = trainer.evaluate(eval_dataset=ds["test"], metric_key_prefix="test")
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)
        
        print(f"Test Results:")
        print(f"WER: {test_metrics.get('test_wer', 'N/A')}%")
        print(f"CER: {test_metrics.get('test_cer', 'N/A')}%")
    
    # MANUAL WER VERIFICATION
    print("MANUAL WER VERIFICATION")
    
    try:
        # Install jiwer if not available
        try:
            from jiwer import wer as jiwer_wer, cer as jiwer_cer
        except ImportError:
            print("  Installing jiwer...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "jiwer"])
            from jiwer import wer as jiwer_wer, cer as jiwer_cer
        
        # Get predictions on test set
        print("  Computing predictions on test set...")
        test_results = trainer.predict(ds["test"])
        pred_ids = test_results.predictions
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        
        label_ids = test_results.label_ids
        label_ids = np.where(label_ids != -100, label_ids, processor.tokenizer.pad_token_id)
        
        # Decode and normalize
        pred_str = [normalize_vietnamese_text(s) for s in processor.batch_decode(pred_ids, skip_special_tokens=True)]
        label_str = [normalize_vietnamese_text(s) for s in processor.batch_decode(label_ids, skip_special_tokens=True)]
        
        # Filter empty
        valid_pairs = [(p, l) for p, l in zip(pred_str, label_str) if p and l]
        if valid_pairs:
            pred_valid, label_valid = zip(*valid_pairs)
            
            # Calculate metrics using jiwer
            real_wer = jiwer_wer(list(label_valid), list(pred_valid))
            real_cer = jiwer_cer(list(label_valid), list(pred_valid))
            
            print(f"Verified metrics (jiwer library):")
            print(f"WER: {real_wer*100:.2f}%")
            print(f"CER: {real_cer*100:.2f}%")
            
            # Compare with trainer metrics
            trainer_wer = test_metrics.get('test_wer', 'N/A')
            trainer_cer = test_metrics.get('test_cer', 'N/A')
            print(f"Comparison with trainer metrics:")
            print(f"Trainer WER: {trainer_wer}%")
            print(f"jiwer WER: {real_wer*100:.2f}%")
            print(f"Trainer CER: {trainer_cer}%")
            print(f"jiwer CER: {real_cer*100:.2f}%")
            
            # Show sample predictions with normalization
            print(f"Sample predictions (5 random, normalized):")
            sample_indices = random.sample(range(len(valid_pairs)), min(5, len(valid_pairs)))
            
            for idx, i in enumerate(sample_indices, 1):
                p, l = valid_pairs[i]
                print(f"[{idx}]")
                print(f"Pred: {p[:90]}{'...' if len(p) > 90 else ''}")
                print(f"True: {l[:90]}{'...' if len(l) > 90 else ''}")
                
                # Per-sample metrics
                try:
                    sample_wer = jiwer_wer([l], [p])
                    sample_cer = jiwer_cer([l], [p])
                    
                    # Word-level comparison
                    pred_words = p.split()
                    true_words = l.split()
                    common_words = sum(1 for pw, tw in zip(pred_words, true_words) if pw == tw)
                    
                    print(f"WER: {sample_wer*100:.1f}% | CER: {sample_cer*100:.1f}%")
                    print(f"Words: {len(pred_words)} pred, {len(true_words)} true, {common_words} match")
                except Exception as e:
                    print(f"Error: {e}")
            
            # Statistics
            print(f"Dataset statistics:")
            print(f"Total samples: {len(pred_str)}")
            print(f"Valid samples: {len(valid_pairs)}")
            print(f"Empty predictions: {len([p for p in pred_str if not p])}")
            print(f"Avg pred length: {np.mean([len(p.split()) for p in pred_valid]):.1f} words")
            print(f"Avg true length: {np.mean([len(l.split()) for l in label_valid]):.1f} words")
            print(f"Avg pred chars: {np.mean([len(p.replace(' ', '')) for p in pred_valid]):.1f}")
            print(f"Avg true chars: {np.mean([len(l.replace(' ', '')) for l in label_valid]):.1f}")
            
            # WER/CER distribution
            wer_scores = []
            cer_scores = []
            for p, l in valid_pairs[:100]:  # Sample 100 for speed
                try:
                    wer_scores.append(jiwer_wer([l], [p]))
                    cer_scores.append(jiwer_cer([l], [p]))
                except:
                    pass
            
            if wer_scores:
                print(f"Error distribution (100 samples):")
                print(f"WER - Min: {min(wer_scores)*100:.1f}%, Max: {max(wer_scores)*100:.1f}%, Median: {np.median(wer_scores)*100:.1f}%")
                print(f"CER - Min: {min(cer_scores)*100:.1f}%, Max: {max(cer_scores)*100:.1f}%, Median: {np.median(cer_scores)*100:.1f}%")
            
        else:
            print("No valid predictions found")
            
    except Exception as e:
        import traceback
        print(f"Verification failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        print("Note: Using CER as the primary metric for Vietnamese ASR.")
    
    print("TRAINING COMPLETE!")
    print(f"Model location: {OUTPUT_DIR}")
    print(f"Final validation WER: {eval_metrics.get('eval_wer', 'N/A')}%")
    print(f"Final validation CER: {eval_metrics.get('eval_cer', 'N/A')}%")
