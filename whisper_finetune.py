import os
import torch
import time
import io
import soundfile as sf
import evaluate
from dataclasses import dataclass
from typing import Any
from datasets import load_dataset, Audio, Dataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

# Setup environment
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/tmp/hf_home"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Auth
user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_TOKEN")
login(token=hf_token)

# Load Dataset
ds = load_dataset("YuvrajGujari/balti-tarjuman-data")
ds = ds.cast_column("audio", Audio(decode=False))

def is_readable(example):
    try:
        sf.read(io.BytesIO(example["audio"]["bytes"]), frames=1)
        return True
    except Exception:
        return False

ds = ds.filter(is_readable, num_proc=4)

# Load Cold-Start Base Model
model_name = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)

model.config.forced_decoder_ids = None
model.generation_config.forced_decoder_ids = None

# SpecAugment setup
model.config.apply_spec_augment = True
model.config.mask_time_prob = 0.05
model.config.mask_feature_prob = 0.05

# Generator setup
def make_generator(split_ds, split_name):
    def gen():
        total = len(split_ds)
        for idx in range(total):
            example = split_ds[idx]
            audio_array, sr = sf.read(io.BytesIO(example["audio"]["bytes"]))
            if sr != 16000:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
                sr = 16000
            input_features = processor.feature_extractor(audio_array, sampling_rate=sr).input_features[0]
            labels = processor.tokenizer(example["text"]).input_ids
            yield {"input_features": input_features, "labels": labels}
    return gen

train_dataset = Dataset.from_generator(make_generator(ds["train"], "train"))
validation_dataset = Dataset.from_generator(make_generator(ds["validation"], "validation"))

# Collator
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

# Metric
metric = evaluate.load("wer")
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    return {"wer": 100 * metric.compute(predictions=pred_str, references=label_str)}

# Safe Training Arguments (Max 2500 steps to capture the peak)
training_args = Seq2SeqTrainingArguments(
    output_dir="/kaggle/working/whisper-small-balti-coldstart",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    warmup_steps=500,
    max_steps=2500,                    # Targets the Step 2000-2500 peak range
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=500,
    save_steps=500,
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    logging_steps=50,
    report_to=[],
    
    # --- CRITICAL DISK SAFETY FLAGS ---
    save_only_model=True,              # Eliminates 1.8GB optimizer file saves
    save_total_limit=2,                # Automatically deletes older checkpoints
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
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

# Launch
trainer.train()
trainer.push_to_hub()
