# %%
from dotenv import load_dotenv
import os
load_dotenv()
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"])

# %%
from datasets import load_dataset

# BOUQuET's bft_Arab config — small, curated English-Balti pairs
ds = load_dataset("facebook/bouquet", "bft_Arab")
print(ds)
print(ds["dev"][0])   # inspect actual column names/structure before assuming

# %%
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# %%
# NLLB needs explicit src/tgt language codes. bft_Arab isn't in NLLB's vocab,
# so we add it as a new special token and let fine-tuning learn its embedding.
# Confirm this is genuinely needed by checking the tokenizer's known langs first.
print("bft_Arab" in tokenizer.additional_special_tokens if hasattr(tokenizer, "additional_special_tokens") else "check needed")

# %%
def prepare_batch(example):
    # Adjust field names once cell 2's actual output confirms them (likely "eng" / "bft_Arab" or similar)
    inputs = tokenizer(example["bft_Arab"], text_target=example["eng"], return_tensors=None, truncation=True, max_length=128)
    return inputs

tokenized = ds.map(prepare_batch, remove_columns=ds["dev"].column_names)

# %%
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
bleu = evaluate.load("sacrebleu")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    result = bleu.compute(predictions=pred_str, references=[[l] for l in label_str])
    return {"bleu": result["score"]}

# %%
training_args = Seq2SeqTrainingArguments(
    output_dir="./nllb-balti-mt",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=3e-5,
    warmup_steps=20,
    max_steps=300,          # small dataset — deliberately modest step count, watch for overfitting early
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    save_steps=50,
    eval_steps=50,
    logging_steps=10,
    report_to=[],
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    greater_is_better=True,
    push_to_hub=True,
    hub_model_id="YuvrajGujari/nllb-balti-mt",
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=tokenized["dev"],   # dev=train per your notes (504 rows)
    eval_dataset=tokenized["test"],   # test=eval (854 rows)
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=tokenizer,
)

# %%
trainer.train()

# %%
trainer.push_to_hub()