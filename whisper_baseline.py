# %%
import os
os.environ["HF_HOME"] = "/tmp/hf_home"
os.environ["HF_HUB_CACHE"] = "/tmp/hf_hub_cache"
os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_datasets_cache"
os.environ["HF_HUB_DISABLE_XET"] = "1"
# NOW import dotenv/login — after env vars, not before
from dotenv import load_dotenv
load_dotenv()
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"])

# %%
from transformers import WhisperForConditionalGeneration, WhisperProcessor
model_name = "openai/whisper-small"
model = WhisperForConditionalGeneration.from_pretrained(model_name)
processor = WhisperProcessor.from_pretrained(model_name)

# %%
import shutil
# nuke the cache fully first — rules out a corrupted-but-"complete" metadata.csv
for d in ["/tmp/hf_datasets_cache", "/tmp/hf_hub_cache", "/tmp/hf_home"]:
    shutil.rmtree(d, ignore_errors=True)

# %%
from huggingface_hub import snapshot_download
import time

local_dir = None
for attempt in range(30):
    try:
        local_dir = snapshot_download(
            repo_id="NasuAhmed/balti-voice-dataset",
            repo_type="dataset",
            local_dir="/tmp/balti_raw",
            max_workers=2,
        )
        break
    except Exception as e:
        wait = min(60 * (attempt + 1), 300)
        print(f"attempt {attempt+1} failed ({type(e).__name__}), retrying in {wait}s...")
        time.sleep(wait)

print(local_dir)

# %%
import os, pandas as pd

meta = pd.read_csv(os.path.join(local_dir, "metadata.csv"))
print("metadata rows:", len(meta))

audio_dir = local_dir  # adjust if mp3s are in a subfolder — check with os.listdir(local_dir) first
on_disk = {f for f in os.listdir(audio_dir) if f.endswith(".mp3")}
print("mp3 files on disk:", len(on_disk))

meta_files = set(meta["file_name"])
missing_metadata = on_disk - meta_files      # audio with no metadata row (this is what crashed you)
missing_audio = meta_files - on_disk          # metadata rows with no audio (incomplete download)

print("audio files with no metadata:", len(missing_metadata))
print("metadata rows with no audio file:", len(missing_audio))

# inner join — keep only rows where both exist
meta_clean = meta[meta["file_name"].isin(on_disk)].reset_index(drop=True)
print("usable pairs:", len(meta_clean))

# %%
from datasets import Dataset, Audio

meta_clean["audio"] = meta_clean["file_name"].apply(lambda f: os.path.join(audio_dir, f))
ds = Dataset.from_pandas(meta_clean[["audio", "sentence"]])
ds = ds.cast_column("audio", Audio(sampling_rate=16000))
print(ds)
# %%
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

sample_ds = ds  # already flat, no split — cast_column already applied earlier if you did it there

for i in range(5):
    sample = sample_ds[i]
    input_features = processor(sample["audio"]["array"], sampling_rate=16000, return_tensors="pt").input_features.to(device)
    predicted_ids = model.generate(input_features)
    prediction = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    reference = sample["sentence"]
    print(f"--- Example {i} ---")
    print(f"REF:  {reference}")
    print(f"PRED: {prediction}")
    print()
# %%
