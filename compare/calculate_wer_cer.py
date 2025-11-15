import pandas as pd
import os
import soundfile as sf
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from jiwer import wer, cer
import re
import torchaudio
csv_path = "/content/metadata.csv"
audio_folder = "/content/audio"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoProcessor.from_pretrained("vinai/PhoWhisper-small")
model = AutoModelForSpeechSeq2Seq.from_pretrained("vinai/PhoWhisper-small").to(device)
model.eval()
df = pd.read_csv(csv_path)
def normalize_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text
results = []
for idx, row in df.iterrows():
    audio_path = row['audio_path']
    gt_text = row['text']

    if not os.path.exists(audio_path):
        print(f"File {audio_path} not exits")
        continue
    audio_input, sr = sf.read(audio_path)
    if len(audio_input.shape) > 1: 
        audio_input = audio_input.mean(axis=1)
    if sr != 16000:
        audio_tensor = torch.from_numpy(audio_input)
        audio_input = torchaudio.functional.resample(audio_tensor, orig_freq=sr, new_freq=16000).numpy()
        sr = 16000
    inputs = processor(audio_input, sampling_rate=sr, return_tensors="pt")
    input_features = inputs["input_features"].to(device)
    with torch.no_grad():
        generated_tokens = model.generate(input_features)
    transcription = processor.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    gt_norm = normalize_text(gt_text)
    hyp_norm = normalize_text(transcription)

    wer_score = wer(gt_norm, hyp_norm)
    cer_score = cer(gt_norm, hyp_norm)

    results.append({
        "audio_path": audio_path,
        "ground_truth": gt_text,
        "transcription": transcription,
        "WER": wer_score,
        "CER": cer_score
    })

    print(f"{os.path.basename(audio_path)}: WER={wer_score*100:.2f}%, CER={cer_score*100:.2f}%")

result_df = pd.DataFrame(results)
result_df.to_csv("asr_results.csv", index=False)
avg_wer = result_df["WER"].mean()
avg_cer = result_df["CER"].mean()
print(f"Avergare WER: {avg_wer*100:.2f}%")
print(f"Avergare CER: {avg_cer*100:.2f}%")
