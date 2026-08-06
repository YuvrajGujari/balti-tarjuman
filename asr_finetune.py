# %%
import os
os.environ["HF_HOME"] = "/tmp/hf_home"
os.environ["HF_HUB_CACHE"] = "/tmp/hf_hub_cache"
os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_datasets_cache"
os.environ["HF_HUB_DISABLE_XET"] = "1"

from dotenv import load_dotenv
load_dotenv()
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"])

# %%
from datasets import load_dataset, Audio

ds = load_dataset("YuvrajGujari/balti-tarjuman-data")
ds = ds.cast_column("audio", Audio(decode=False))

# %%
import soundfile as sf
import io

def is_readable(example):
    try:
        sf.read(io.BytesIO(example["audio"]["bytes"]), frames=1)
        return True
    except Exception:
        return False

print("Checking for corrupted audio files...")
ds = ds.filter(is_readable, num_proc=4)
print({split: len(ds[split]) for split in ds})

# %%
from transformers import WhisperForConditionalGeneration, WhisperProcessor

model_name = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# %%
import time

sample = ds["train"][0]
print("Got sample, bytes length:", len(sample["audio"]["bytes"]))

start = time.time()
audio_array, sr = sf.read(io.BytesIO(sample["audio"]["bytes"]))
print(f"sf.read took {time.time()-start:.2f}s, shape={audio_array.shape}, sr={sr}")

if sr != 16000:
    import librosa
    start = time.time()
    audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
    sr = 16000
    print(f"resample took {time.time()-start:.2f}s")

start = time.time()
features = processor.feature_extractor(audio_array, sampling_rate=sr).input_features[0]
print(f"feature_extractor took {time.time()-start:.2f}s")

start = time.time()
labels = processor.tokenizer(sample["text"]).input_ids
print(f"tokenizer took {time.time()-start:.2f}s")

# %%
import time

def make_generator(split_ds, split_name):
    def gen():
        total = len(split_ds)
        start_time = time.time()
        for idx in range(total):
            example = split_ds[idx]
            audio_array, sr = sf.read(io.BytesIO(example["audio"]["bytes"]))
            if sr != 16000:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
                sr = 16000
            input_features = processor.feature_extractor(audio_array, sampling_rate=sr).input_features[0]
            labels = processor.tokenizer(example["text"]).input_ids
            if idx % 50 == 0:
                elapsed = time.time() - start_time
                print(f"[{split_name}] {idx}/{total} done, {elapsed:.1f}s elapsed", flush=True)
            yield {"input_features": input_features, "labels": labels}
    return gen

# %%
from datasets import Dataset

train_dataset = Dataset.from_generator(make_generator(ds["train"], "train"))
validation_dataset = Dataset.from_generator(make_generator(ds["validation"], "validation"))

print(train_dataset)
print(validation_dataset)
# %%
import torch
from dataclasses import dataclass
from typing import Any

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# %%
import evaluate
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# %%
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-balti",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=100,
    max_steps=1000,
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=100,
    eval_steps=100,
    logging_steps=25,
    report_to=[],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
    hub_model_id="YuvrajGujari/whisper-small-balti",
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor,
)

# %%
trainer.train()

# %%
trainer.push_to_hub()