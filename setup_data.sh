# # Cài thư viện cần thiết
# pip install -q huggingface_hub datasets --upgrade

# # Tạo thư mục lưu data
# mkdir -p ~/datasets/vimddata
# cd ~/datasets/vimddata

# # Tải toàn bộ dữ liệu ViMD_Dataset từ Hugging Face
# huggingface-cli download nguyendv02/ViMD_Dataset --repo-type dataset --local-dir ./ViMD_Dataset

# echo "✅ Dataset downloaded successfully to: ~/datasets/vimddata/ViMD_Dataset"

#!/bin/bash
set -e

echo "=============================="
echo "🚀 BẮT ĐẦU CÀI MÔI TRƯỜNG VIETNAMESE ASR"
echo "=============================="

# === 1. CẬP NHẬT VÀ CÀI ĐẶT CƠ BẢN ===
apt-get update -y && apt-get upgrade -y
apt-get install -y git wget unzip curl ffmpeg build-essential gdown python3-pip

# === 2. CẬP NHẬT PIP ===
python3 -m pip install --upgrade pip setuptools wheel

# === 3. KIỂM TRA GPU ===
echo "🔧 Kiểm tra GPU..."
if command -v nvidia-smi &> /dev/null; then
  nvidia-smi
  echo "✅ GPU sẵn sàng!"
else
  echo "⚠️ Không phát hiện GPU (sẽ train bằng CPU, chậm hơn)"
fi

# === 4. CÀI CÁC THƯ VIỆN YÊU CẦU ===
echo "📦 Cài đặt dependencies..."
pip install --no-cache-dir -r requirement.txt

# === 5. CẤU HÌNH WANDB VÀ HUGGINGFACE CACHE ===
export WANDB_API_KEY="e896b413a2e8bfb2509f89a27cf130ab7dd54840"
export WANDB_PROJECT="phowhisper-vietnamese-accents"
export HF_HOME="/root/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p $TRANSFORMERS_CACHE $HF_DATASETS_CACHE

# === 6. KIỂM TRA PHIÊN BẢN ===
echo "=============================="
echo "✅ Môi trường đã sẵn sàng!"
echo "Python: $(python3 --version)"
echo "Torch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo "=============================="

echo ""
echo "🎯 Tiếp theo: tải dataset bằng lệnh"
echo "bash setup_data.sh"
