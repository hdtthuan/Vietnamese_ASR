import os, shutil
from transformers import WhisperForConditionalGeneration
from ctranslate2.converters import TransformersConverter

checkpoint_dir = "model_checkpoint-4600"
ctranslate2_dir = "fine_tune_model"

# Load model from checkpoint Hugging Face
model = WhisperForConditionalGeneration.from_pretrained(checkpoint_dir)

# Convert into base CTranslate2
converter = TransformersConverter(
    model_name_or_path=checkpoint_dir,
    copy_files=[
        "vocab.json",
        "merges.txt",
        "tokenizer_config.json",
        "generation_config.json",
        "preprocessor_config.json",
    ],
)
converter.convert(ctranslate2_dir, force=True)
print("Model has been converted to:", ctranslate2_dir)