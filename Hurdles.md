Balti Tarjuman — Hurdles Log
1. HF cache PermissionError (initial diagnosis — later revised)

Symptom: PermissionError on .cache/huggingface, files created with ---------- permissions regardless of umask. Initial theory: The /teamspace/studios/this_studio mount is a custom "lightning" filesystem type that creates new files with zero permissions. Fix attempted: Set HF_HOME / HF_HUB_CACHE / HF_DATASETS_CACHE to /home/zeus/... instead of the /teamspace mount, as the very first cell before any HF library import (env vars are locked in at import time by huggingface_hub/datasets). Caveat: This fix requires a full kernel restart plus correct cell ordering — setting the env vars after an earlier login() or model-load cell doesn't work, even in the same session.

2. Same PermissionError recurs across /teamspace, /home/zeus, AND /tmp

Revised diagnosis: Not a filesystem/mount issue at all. HF's rate-limit-interrupted downloads leave stale .incomplete files (hash-named per URL) with broken permissions. Retries hit the same filename and inherit the broken state. Fix: find <cache_dir> -name "*.incomplete" -delete before every retry, and/or switch from datasets.load_dataset to huggingface_hub.snapshot_download for flat-file, many-item datasets (better resume handling).

3. HTTP 429 rate limiting on 9,481-file flat-file dataset download

Symptom: datasets.load_dataset downloading 9,481 individual mp3 files (no bundled archive) triggers repeated 429 Too Many Requests. Fix: Not a code bug — let the library's backoff retry naturally, with num_proc=1. Patience required; can take a long time for this many files.

4. ValueError: audio ... doesn't have metadata — folder-based builder crash

Symptom: datasets' audiofolder builder crashed because an audio file (common_voice_bft_41845763.mp3) had no matching row in the downloaded metadata.csv. Root cause: Ambiguous — either a truncated/corrupted metadata.csv download (no .incomplete marker to catch it, since the file "completed" with bad/partial content), or genuine upstream data noise (audio files without metadata rows). Fix: Bypassed the strict audiofolder matcher entirely. Used snapshot_download to pull the raw repo, then did a manual pandas inner join between metadata.csv and the actual files on disk, dropping unmatched entries instead of crashing. Built the final Dataset via Dataset.from_pandas + .cast_column("audio", Audio(...)). Result: 9,481 metadata rows, 9,981 mp3s on disk, 500 audio files with no metadata (dropped), 0 metadata rows missing audio → 9,481 clean usable pairs.

5. Xet-backend 429s during snapshot_download

Symptom: ConnectionError: 429 specifically on .../xet-read-token/... endpoints — a separate rate-limit pool from regular file downloads. Fix: Set HF_HUB_DISABLE_XET=1 (before any huggingface_hub import, same ordering constraint as #1) to force the classic download path. Combined with a manual retry loop (30 attempts, capped exponential backoff) around snapshot_download, using local_dir (which resumes cleanly, unlike hash-keyed cache dirs) so partial progress was never lost across retries.

6. Leftover split-based code after switching to a flat Dataset

Symptom: AttributeError: 'Dataset' object has no attribute 'keys'. Root cause: Inference code was still written for the original load_dataset result (which had train/test splits); the new Dataset.from_pandas output has no splits, just flat rows. Fix: Replaced split_name = list(ds.keys())[0]; sample_ds = ds[split_name] with sample_ds = ds directly.

7. Zero-shot Whisper-small baseline — expected-bad output (not a bug)

Result: Zero-shot predictions were garbage relative to Balti references (Khmer-, French-, Armenian-, Hebrew-, and Chinese-looking hallucinated text). This is expected — establishes the "before" number ahead of fine-tuning, consistent with the Pashto/Tarjuman Phase 1 pattern. Cross-checked against a published BaltiVoice paper using the same Common Voice–derived corpus: reported zero-shot baseline ~159% WER, fine-tuned Whisper-small ~26.74% WER — useful ballpark for sanity-checking future numbers.

8. Processed training data never wired into the Trainer

Symptom: ValueError: No columns in the dataset match the model's forward method signature ... The following columns have been ignored: [audio, text]. Root cause: process_split() built train_processed (a plain Python list with input_features/labels), but Seq2SeqTrainer was still pointed at ds["train"] — the original raw, unprocessed dataset. The processed data was computed and never used. Fix: Added the missing validation_processed step (only train had been processed), converted both to Dataset objects, and pointed the trainer at train_dataset/validation_dataset instead of ds["train"]/ds["validation"].

9. Kernel crash (OOM) building Dataset.from_list

Symptom: Kernel crashed silently (no Python traceback) while running Dataset.from_list(train_processed). Root cause: train_processed held ~8,058 full input_features arrays (~940KB each at float32) as a Python list in RAM — roughly 7.5GB+ before conversion even started, and from_list converts row-by-row, spiking memory further. Instance RAM was exceeded, OS OOM-killed the kernel. Fix: Switched to Dataset.from_generator, which streams examples to Arrow-on-disk incrementally instead of materializing the full list in memory first. Folded the old process_split + from_list steps into a single generator-based make_generator() function. Kernel restart recommended before retry, in case OOM left CUDA context in a bad state.

Key transferable lesson

Any HF HF_HOME / HF_HUB_CACHE / HF_DATASETS_CACHE / HF_HUB_DISABLE_XET env var fix must be set before any import of huggingface_hub, datasets, or transformers in that kernel session — including indirect imports via login() or model loading in earlier cells. A kernel restart plus correct cell ordering is required for the fix to actually take effect.
