# 🇻🇳 Vietnamese ASR – Dialect-Aware Speech Recognition

Fine-tuned Whisper-based model on **ViMD Dataset (63 provinces, 3 dialects)**

---

## 📌 Overview

This repository contains the complete pipeline for building a **Vietnamese Automatic Speech Recognition (ASR)** system specialized for **regional dialects**.
The project includes:

* 🔧 **Full preprocessing + training pipeline** for fine-tuning Whisper/PhoWhisper
* 🧪 **Evaluation framework** (coming in the next folder)
* 🖥️ **Streamlit demo UI** for quick inference
* 📦 **Model conversion utilities** for deployment (CT2 / ONNX / HuggingFace format)
* 🚀 Ready-to-run scripts for VastAI, Google Drive, and local machines

This project is built for the **FPT University DSP391m Capstone**, with a strong focus on real-world ASR performance across dialects.

---

## 📁 Repository Structure

```
Vietnamese_ASR/
│
├── demo/                     # Streamlit demo interface
│   └── demo.py
│
├── fine_tune_model/          # This folder includes model weights and tokenizers, há to be downloaded from Google Drive
│   └── (copy model files from Google Drive here)
│
├── evaluation/               # (Sẽ thêm) Evaluation scripts for comparing models
│   └── ...
│
├── convert_model.py          # Convert model → CT2, ONNX, HF format
├── train.py                  # Training / fine-tuning script
│
├── setup.sh                  # Environment setup for VastAI / Linux
├── setup_data.sh             # Download + extract processed ViMD dataset
│
├── requirement.txt
└── README.md                 # (this file)
```

---

## 🔧 Installation

### 1️⃣ Clone repo

```bash
git clone https://github.com/<your_repo>/Vietnamese_ASR.git
cd Vietnamese_ASR
```

### 2️⃣ Create environment

Use conda or venv:

```bash
bash setup.sh
```

Or manually:

```bash
pip install -r requirement.txt
```

---

## 📥 Prepare Model Files

Your teammate provides a Google Drive folder containing:

```
train_outputs/
└── phowhisper_vimd/
    └── ctranslate2_model/
```

Copy toàn bộ files trong `ctranslate2_model/` vào:

```
Vietnamese_ASR/fine_tune_model/
```

---

## 🎧 Streamlit Demo

### 1️⃣ Go to demo folder

```bash
cd demo
```

### 2️⃣ Run demo

```bash
streamlit run demo.py
```

Sau đó truy cập:
👉 [http://localhost:8501](http://localhost:8501)

---

## 🏋️ Training

### 1️⃣ Prepare dataset

Processed ViMD dataset stored on Google Drive.

Run:

```bash
bash setup_data.sh
```

This script will:

* Mount or download from Google Drive
* Extract dataset
* Organize into `train/` – `valid/` – `test/` folders

### 2️⃣ Start fine-tuning

```bash
python train.py --config configs/vimd_config.yaml
```

Training script includes:

* Augmentation
* Mixed precision
* Gradient accumulation
* Checkpoint saving
* Logging (loss, WER, CER)

---

## 🔄 Model Conversion

To convert the fine-tuned model into **CTranslate2** for fast inference:

```bash
python convert_model.py --source <path_to_model> --output fine_tune_model/
```

Supports:

* CTranslate2
* HuggingFace
* ONNX (coming soon)

---

## 🧪 Evaluation (Upcoming Folder)

A new folder `/evaluation` will contain:

* 📊 Compare Whisper base vs large vs PhoWhisper vs your fine-tuned model
* 🏷️ Evaluate per dialect: North / Central / South
* 🏅 Compute WER / CER / Speaker-level performance
* 🔉 Noise robustness evaluation
* 📈 Visualizations (confusion matrix, error samples)

Example (coming soon):

```
evaluation/
│   evaluate_ct2.py
│   evaluate_hf.py
│   compare_models.ipynb
│   dialect_breakdown.csv
```

---

## 🧠 Model Details

* Base model: **PhoWhisper** (Vietnamese-specialized Whisper variant)
* Fine-tuning dataset: **ViMD – 102.5 hours – 63 provinces**
* Tokenizer: SentencePiece
* Feature extractor: 80-channel Mel-spectrogram
* Optimizer: AdamW
* Metrics: WER / CER (character-level suited for Vietnamese)

---

## 🗂 Dataset

We use **ViMD**, a large-scale Vietnamese dialect dataset:

| Region  | Provinces | %   |
| ------- | --------- | --- |
| North   | 25        | 40% |
| Central | 19        | 30% |
| South   | 19        | 30% |

Includes:

* 1.5M text characters
* 80k+ spoken utterances
* Natural speech (non-studio)
* Full demographic metadata

---

## 🚀 Deployment (Future Work)

Planned additions:

* FastAPI real-time ASR server
* gRPC service
* Mobile-ready model export
* Websocket streaming

---

## 🤝 Contributors

* **Thuận Hoàng** – AI Engineer
* **Khoa Châu** – Model Training / Demo
* **ViMD Team** – Dataset providers
* FPT University – Faculty of AI & DS

---

## 📄 License

MIT License
(Feel free to use, modify, and cite our work.)

---

## 📬 Contact

For questions or collaboration:

📧 **[kodtt1234@gmail.com](mailto:kodtt1234@gmail.com)**

---

Nếu bạn muốn, tôi có thể thêm:

✅ Badges (Python version, license, model size, WER score)
✅ Thêm hình minh họa kiến trúc Whisper
✅ Banner đẹp cho GitHub
✅ Tạo “demo video” hướng dẫn trong README

Bạn muốn mở rộng README theo hướng nào?
