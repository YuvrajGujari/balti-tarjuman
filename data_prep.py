# %%Rebuild meta_clean (skip re-download — files already at /tmp/balti_raw)
import os, pandas as pd
os.environ["HF_HOME"] = "/tmp/hf_home"
os.environ["HF_HUB_CACHE"] = "/tmp/hf_hub_cache"
os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_datasets_cache"
os.environ["HF_HUB_DISABLE_XET"] = "1"
from dotenv import load_dotenv
load_dotenv()
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"])

local_dir = "/tmp/balti_raw"
meta = pd.read_csv(os.path.join(local_dir, "metadata.csv"))
audio_dir = local_dir
on_disk = {f for f in os.listdir(audio_dir) if f.endswith(".mp3")}
meta_clean = meta[meta["file_name"].isin(on_disk)].reset_index(drop=True)
print("usable pairs:", len(meta_clean))

# %%Build the Dataset object
from datasets import Dataset, Audio
meta_clean["audio"] = meta_clean["file_name"].apply(lambda f: os.path.join(audio_dir, f))
ds = Dataset.from_pandas(meta_clean[["audio", "sentence"]])
ds = ds.cast_column("audio", Audio(sampling_rate=16000))
print(ds)

# %%
import soundfile as sf

def get_duration(path):
    info = sf.info(path)
    return info.frames / info.samplerate

meta_clean["duration"] = meta_clean["audio"].apply(get_duration)
meta_clean_filtered = meta_clean[
    (meta_clean["sentence"].str.strip().str.len() > 0) &
    (meta_clean["duration"] >= 0.5) &
    (meta_clean["duration"] <= 30)
].reset_index(drop=True)

print(f"Before cleaning: {len(meta_clean)} | After cleaning: {len(meta_clean_filtered)}")

# Now build the Dataset from the already-filtered dataframe
from datasets import Dataset, Audio
ds_clean = Dataset.from_pandas(meta_clean_filtered[["audio", "sentence"]])
ds_clean = ds_clean.cast_column("audio", Audio(sampling_rate=16000))
print(ds_clean)

# %%Now the rename + split + push steps you already had
ds_clean = ds_clean.rename_column("sentence", "text")

split1 = ds_clean.train_test_split(test_size=0.15, seed=42)
split2 = split1["test"].train_test_split(test_size=0.33, seed=42)
# %%
from datasets import DatasetDict
final = DatasetDict({
    "train": split1["train"],
    "validation": split2["train"],
    "test": split2["test"],
})
print({k: len(v) for k, v in final.items()})

final.push_to_hub("YuvrajGujari/balti-tarjuman-data", private=True)
print("Pushed to Hugging Face Hub.")
# %%
print({k: len(v) for k, v in final.items()})