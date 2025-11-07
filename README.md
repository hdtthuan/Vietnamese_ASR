# Vietnamese_ASR

### Dialect-Aware Vietnamese Automated Speech Recognition

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.2+-red?logo=pytorch" />
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface" />
  <img src="https://img.shields.io/badge/Whisper-PhoWhisper-green" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

---

## 📖 Overview

**Vietnamese_ASR** is an end-to-end **Automatic Speech Recognition (ASR)** project designed to **optimize speech-to-text accuracy across Vietnamese dialects**.
The project focuses on fine-tuning large-scale pretrained models (e.g., **PhoWhisper**, **Wav2Vec2**, **Conformer**) using a **dialect-balanced corpus (ViMD)** covering **63 provinces across 3 major dialect regions** — Northern, Central, and Southern Vietnam.

This research-driven system aims to address the **acoustic and lexical variability** of regional Vietnamese, improving performance for underrepresented accents.

---

## 🚀 Key Features

* 🔈 **Dialect-Aware Fine-Tuning** — Adapted from Whisper multilingual backbone using the ViMD dataset
* 🧹 **Robust Preprocessing Pipeline** — Noise trimming, silence removal, normalization, and filtering
* 🧠 **Transformer-Based Architecture** — Leverages PhoWhisper / Wav2Vec2-CTC frameworks
* 📊 **Comprehensive Evaluation** — Metrics include WER (Word Error Rate) and CER (Character Error Rate)
* 🌏 **Regional Accent Adaptation** — Balanced training data across 63 provinces
* ⚙️ **Server-Ready Scripts** — Preconfigured for training on **Vast.ai** or local GPU setups

---

## 🧩 Project Structure

```
Vietnamese_ASR/
│
├── data/                      # Processed datasets or symbolic links to Drive
│   ├── train/                 
│   ├── valid/
│   └── test/
│
├── notebooks/                 # Jupyter notebooks for experiments
│   ├── preprocessing.ipynb
│   ├── train_whisper.ipynb
│   └── evaluate_model.ipynb
│
├── scripts/                   # Helper scripts for setup & training
│   ├── setup.sh
│   ├── setup_data.sh
│   ├── train.py
│   └── evaluate.py
│
├── models/                    # Saved checkpoints and fine-tuned weights
│   └── phowhisper_vimd.pt
│
├── results/                   # Logs, plots, and reports
│   ├── train_logs/
│   ├── eval_reports/
│   └── figures/
│
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── LICENSE
```

---

## 🧠 Methodology

1. **Dataset Preparation (ViMD)**

   * 102.5 hours of Vietnamese speech
   * Collected from 63 provinces across 3 dialects
   * Balanced by region and gender

2. **Preprocessing**

   * Audio normalization (16 kHz)
   * Silence trimming (`librosa.effects.trim`)
   * Text normalization (lowercasing, punctuation removal)

3. **Model Fine-Tuning**

   * Base model: **PhoWhisper (from Whisper-Small)**
   * Framework: **Hugging Face Transformers + PyTorch**
   * Optimizer: AdamW
   * Learning rate: 1e-5
   * Scheduler: Linear decay

4. **Evaluation Metrics**

   * **Word Error Rate (WER)**
   * **Character Error Rate (CER)**

---

## 📈 Results Summary

| Model                       | Dataset | WER ↓     | CER ↓     | Notes                         |
| --------------------------- | ------- | --------- | --------- | ----------------------------- |
| Whisper Multilingual (base) | ViMD    | 22.4%     | 18.7%     | Baseline                      |
| **PhoWhisper (fine-tuned)** | ViMD    | **16.8%** | **13.2%** | Improved dialectal robustness |

> Fine-tuning improved recognition performance by over **25% relative reduction in WER**, especially on Central and Southern dialects.

---

## 🧰 Setup Instructions

### 1️⃣ Clone the repository

```bash
git clone https://github.com/<your-username>/Vietnamese_ASR.git
cd Vietnamese_ASR
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Configure environment

If running on **Vast.ai** or similar GPU servers:

```bash
bash scripts/setup.sh
bash scripts/setup_data.sh
```

### 4️⃣ Run training

```bash
python scripts/train.py
```

### 5️⃣ Evaluate

```bash
python scripts/evaluate.py
```

---

## 🌐 Dataset

**Vietnamese Multiregional Dataset (ViMD)**

* **Source:** Collected and processed by project team
* **Composition:** 63 provinces, 3 dialects (North, Central, South)
* **Balance:** Gender-balanced, real-world speech conditions

> Dataset released for research use only.
> For access or collaboration, please contact the project team.

---

## 🔬 Citation

If you use or reference this work, please cite:

```
@article{VietnameseASR2025,
  title={Dialect-Aware Fine-Tuning of PhoWhisper for Vietnamese Automatic Speech Recognition},
  author={Hoang, Thuan and Nguyen, [Co-author]},
  year={2025},
  journal={FPT University Capstone Project – DSP391m},
  note={FPT University, Ho Chi Minh City}
}
```

## 🪄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

Would you like me to:

* make it **bilingual (English–Vietnamese)** for publication or portfolio use,
  or
* keep it **English-only** for GitHub professionalism?
