import io
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Union

from datasets import Audio, Dataset, load_dataset
import evaluate
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient
import librosa
import soundfile as sf
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

# 1. Auth Setup
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_TOKEN")
login(token=hf_token)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 2. Dataset Loading
ds = load_dataset("YuvrajGujari/balti-tarjuman-data")
ds = ds.cast_column("audio", Audio(decode=False))

# 3. Vocabulary Construction
def extract_chars(dataset_split):
    vocab = set()
    for text in dataset_split["text"]:
        vocab.update(list(text))
    return vocab

train_vocab = extract_chars(ds["train"])
val_vocab = extract_chars(ds["validation"])
all_chars = sorted(train_vocab | val_vocab)

vocab_dict = {char: idx for idx, char in enumerate(all_chars)}
vocab_dict["|"] = vocab_dict.pop(" ") if " " in vocab_dict else len(vocab_dict)
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)

with open("vocab.json", "w") as f:
    json.dump(vocab_dict, f)

# 4. Tokenizer & Processor Setup
tokenizer = Wav2Vec2CTCTokenizer(
    "vocab.json",
    unk_token="[UNK]",
    pad_token="[PAD]",
    word_delimiter_token="|",
)

feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=True,
)

processor = Wav2Vec2Processor(
    feature_extractor=feature_extractor, tokenizer=tokenizer
)

# 5. Model Initialization
# --- CHANGE vs. the 22.82% WER baseline run: SpecAugment added. ---
# Kept mild (0.05/0.05) since the baseline run showed slow-but-steady
# convergence with no overfitting signal — this is meant to test whether
# augmentation helps generalization on top of that, not to replace a
# training-length fix the way it did for Whisper.
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-xls-r-300m",
    vocab_size=len(processor.tokenizer),
    pad_token_id=processor.tokenizer.pad_token_id,
    ctc_loss_reduction="mean",
    apply_spec_augment=True,
    mask_time_prob=0.05,
    mask_feature_prob=0.05,
).to(device)

model.freeze_feature_encoder()
model.config.use_cache = False

# 6. Data Generator
def make_generator(split_ds, split_name):
    def gen():
        total = len(split_ds)
        start_time = time.time()
        for idx in range(total):
            example = split_ds[idx]
            audio_array, sr = sf.read(io.BytesIO(example["audio"]["bytes"]))

            if sr != 16000:
                audio_array = librosa.resample(
                    audio_array, orig_sr=sr, target_sr=16000
                )
                sr = 16000

            input_values = processor(
                audio_array, sampling_rate=sr
            ).input_values[0]
            labels = processor(text=example["text"]).input_ids

            if idx % 500 == 0:
                elapsed = time.time() - start_time
                print(
                    f"[{split_name}] {idx}/{total} processed ({elapsed:.1f}s)",
                    flush=True,
                )

            yield {"input_values": input_values, "labels": labels}

    return gen

train_dataset = Dataset.from_generator(make_generator(ds["train"], "train"))
validation_dataset = Dataset.from_generator(
    make_generator(ds["validation"], "validation")
)

# 7. Data Collator
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(
            input_features, padding=self.padding, return_tensors="pt"
        )
        labels_batch = self.processor.tokenizer.pad(
            label_features, padding=self.padding, return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

# 8. Evaluation Metrics
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = torch.argmax(torch.tensor(pred_logits), dim=-1)

    label_ids = pred.label_ids.copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# 9. Training Arguments & Trainer Setup
# --- CHANGE vs. baseline: LR nudged down slightly (3e-4 -> 2e-4) to pair
# with the added augmentation. Step budget UNCHANGED (still 15 epochs,
# ~7,400 steps) — the baseline's WER curve was still declining at step
# 7400 with no plateau, so cutting the budget the way the Whisper recipe
# did would undertrain this model rather than regularize it. ---
training_args = TrainingArguments(
    output_dir="./wav2vec2-balti-specaugment",
    hub_model_id="YuvrajGujari/wav2vec2-balti-specaugment",
    group_by_length=False,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    eval_strategy="steps",
    eval_steps=200,
    save_steps=200,
    logging_steps=25,
    learning_rate=2e-4,
    warmup_steps=300,
    num_train_epochs=15,
    fp16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    save_total_limit=3,
    metric_for_best_model="wer",
    greater_is_better=False,
    load_best_model_at_end=True,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor,
)

# 10. Execute Training & Push Artifacts
trainer.train()
trainer.push_to_hub(commit_message="Upload Wav2Vec2 Balti + SpecAugment fine-tuned model")
