# %%
# === MT Model Discovery — investigating translation options for Balti (bft) ===
# Goal: find a usable Balti-to-English (or English-to-Balti with reversible direction) MT model/data

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import hf_hub_download, list_repo_files
import json

# %%
# --- Candidate 1: LiaqatAlisha/english-balti-translation ---
tokenizer = AutoTokenizer.from_pretrained("LiaqatAlisha/english-balti-translation")
model = AutoModelForSeq2SeqLM.from_pretrained("LiaqatAlisha/english-balti-translation")

text = "Hello, how are you today?"
inputs = tokenizer(text, return_tensors="pt")
generated = model.generate(**inputs, max_length=50)
print(tokenizer.batch_decode(generated, skip_special_tokens=True))
# Result: English -> Balti direction, but output was in TIBETAN script,
# not the Nastaliq/Perso-Arabic script our dataset uses. Wrong direction + wrong script.

# %%
path = hf_hub_download("LiaqatAlisha/english-balti-translation", "README.md")
print(open(path).read())
# README was minimal (just license: mit), no training data provenance info

# %%
print(list_repo_files("LiaqatAlisha/english-balti-translation"))

# %%
# Note: tokenizer.additional_special_tokens and tokenizer.lang_code_to_id both raise
# AttributeError on this transformers version's NllbTokenizer — had to inspect raw config files instead
path = hf_hub_download("LiaqatAlisha/english-balti-translation", "tokenizer_config.json")
config = json.load(open(path))
bft_tokens = [t for t in config.get("additional_special_tokens", []) if "bft" in str(t).lower()]
print("Balti-related tokens in tokenizer_config:", bft_tokens)
# Result: empty. Confirms this is the full stock NLLB-200 tokenizer, no custom bft code added.

# %%
path = hf_hub_download("LiaqatAlisha/english-balti-translation", "added_tokens.json")
added = json.load(open(path))
print(json.dumps(added, indent=2))
# Result: only a handful of raw Tibetan-script Unicode characters were added as vocab tokens,
# not a formal bft language code. Confirms: no reliable direction/script control on this model.
# CONCLUSION: not usable for this pipeline.

# %%
# --- Candidate 2: facebook/nllb-200-distilled-600M (generic base model) ---
path = hf_hub_download("facebook/nllb-200-distilled-600M", "tokenizer_config.json")
raw_text = open(path).read()
print("bft" in raw_text.lower())

path2 = hf_hub_download("facebook/nllb-200-distilled-600M", "special_tokens_map.json")
raw_text2 = open(path2).read()
print("bft" in raw_text2.lower())
# Result: False, False. Confirmed via raw-text search — bft is not among NLLB-200's 200 languages at all.
# CONCLUSION: no native Balti support in base NLLB-200 either.

# %%
# --- Candidate 3: facebook/bouquet — parallel text dataset (not a model) ---
from datasets import load_dataset, get_dataset_config_names

configs = get_dataset_config_names("facebook/bouquet")
print("bft_Arab" in configs)  # True — confirmed available

bft_ds = load_dataset("facebook/bouquet", "bft_Arab")
print(bft_ds)
print(bft_ds["dev"][0])
# Result: 504 dev + 854 test rows, correct Nastaliq script, high-quality curated pairs
# (src_text = Balti, tgt_text = English). Small but real and usable.
# CONCLUSION: this is the data source used to fine-tune our own MT stage —
# see phase_mt_finetune.py for the actual training run.

# %%
# --- Dataset check for ASR stage (separate from MT — just verifying structure here) ---
files = list_repo_files("NasuAhmed/balti-voice-dataset", repo_type="dataset")
print(files[:30])
# Flat structure: mp3 files at repo root, matching metadata.csv's file_name column directly

ds = load_dataset("NasuAhmed/balti-voice-dataset")
print(ds)
print(ds["train"][0])