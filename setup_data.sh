# Cài thư viện cần thiết
pip install -q huggingface_hub datasets --upgrade

# Tạo thư mục lưu data
mkdir -p ~/datasets/vimddata
cd ~/datasets/vimddata

# Tải toàn bộ dữ liệu ViMD_Dataset từ Hugging Face
huggingface-cli download nguyendv02/ViMD_Dataset --repo-type dataset --local-dir ./ViMD_Dataset

echo "✅ Dataset downloaded successfully to: ~/datasets/vimddata/ViMD_Dataset"