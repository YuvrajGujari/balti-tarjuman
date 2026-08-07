# Balti Tarjuman — Progress Log
Balti-to-English speech-to-speech translation — a personal project building a full pipeline (speech recognition, translation, and speech synthesis) for Balti, a low-resource language with very limited existing digital tooling.

---

## Foundation ✅
- Cloud dev environment set up: Lightning.ai Studio, VS Code Remote-SSH, GitHub repo, Hugging Face account
- Secrets handled via `.env` + `python-dotenv` from the start
- VS Code Auto Save enabled from the start to avoid unsaved-file issues

## First Listen ✅
Zero-shot `whisper-small` on Balti audio: complete failure, as expected for a language the model has no exposure to. Output scattered across unrelated scripts/languages (Khmer, French-like, Armenian-like, Hebrew-like, Chinese characters) — no coherent signal at all. Confirms fine-tuning is necessary, not just helpful.

## Gathering ✅
- **ASR dataset:** `NasuAhmed/balti-voice-dataset` — 9,481 raw rows; hit permission and timeout errors downloading, resolved with a retry loop (exponential backoff) around `snapshot_download`; found and cleaned a metadata/audio mismatch (some files had no metadata row and vice versa) via inner join
- **MT dataset:** resolved after real investigation — NLLB-200 has zero native Balti (`bft`) coverage; found `facebook/bouquet`'s `bft_Arab` config (correct script, English-paired, curated) — small (504 dev + 854 test = 1,358 pairs) but usable and honest to report as-is
- Prior published work exists: "BaltiVoice" paper (arXiv 2606.03504), 26.74% WER Whisper-small ASR — our differentiator is the **full pipeline** (ASR+MT+TTS), not ASR alone

## Voice ✅
Fine-tuned primary ASR (`whisper-small`) on the Balti dataset.
- Manual feature-extraction/tokenization pipeline built (`process_split`), corrupted-audio filtering added (`soundfile` readability check)
- Hit and fixed: processed data never wired into the Trainer — `Seq2SeqTrainer` was still pointed at the raw, unprocessed dataset instead of the processed one
- Hit and fixed: kernel OOM crash converting the full processed dataset to a `Dataset` via `from_list` (~8k examples × full `input_features` arrays exceeded RAM) — switched to `Dataset.from_generator` to stream to disk instead of materializing in memory
- Clean training curve, no overfitting: WER dropped 63.42% → 36.73% over 1000 steps; best checkpoint at step 900 (val loss 0.304, WER 36.73%)
- Pushed to `YuvrajGujari/whisper-small-balti`; local checkpoint-900 weights cross-checked against the pushed weights via the Hugging Face webpage
- Gap vs. published BaltiVoice benchmark (26.74% WER on a similar corpus) noted for the writeup

## Echo ✅
Fine-tuned Wav2Vec2 (`facebook/wav2vec2-xls-r-300m`) on the Balti dataset.
- Custom vocabulary extraction script built directly from character set in `balti-tarjuman-data` (`vocab.json`)
- Streamed audio loading via `Dataset.from_generator` using `soundfile` and `librosa` resampling (16kHz target)
- Built `DataCollatorCTCWithPadding` for dynamic padding and label masking (`-100` target padding)
- **State-of-the-art ASR result achieved:** Completed 15 epochs (7,560 steps); WER steadily declined from 100% → 27.31% (Phase 1, Step 3800) → **22.82% WER** (Phase 2, Step 7400/7560)
- **Benchmark comparison:** Outperformed published *BaltiVoice* baseline (26.74% WER) by **~3.9 percentage points**
- Model weights (`model.safetensors`), processor, and tokenizer uploaded and verified at `YuvrajGujari/wav2vec2-balti`
- Standalone fine-tuning script (`wav2vec2_finetune.py`) committed and pushed to GitHub repo

## Bridge ✅
Fine-tuned `facebook/nllb-200-distilled-600M` on the `bft_Arab` BOUQuET data (504 train, 854 eval pairs) — the project's real novel contribution, as no existing MT solution for Balti exists.
- Added `bft_Arab` as a new special token, embedding warm-started from `bod_Tibt`
- Best checkpoint: step 650, BLEU 4.883 — val loss climbed steadily past step 300 (overfitting, same pattern seen in Whisper Round 2), reported honestly as the real ceiling this data size allows
- Verified via SHA256, pushed to `YuvrajGujari/nllb-balti-mt`; disk cleaned up (non-best checkpoints removed)

## Pipeline Assembly (VAD → ASR → MT → TTS) ✅
- Assembled full end-to-end pipeline: Silero VAD → fine-tuned Wav2Vec2-balti (primary ASR: **22.82% WER**) → MMS-1b-all (pretrained backup ASR) → fine-tuned NLLB-balti-mt (translation) → Kokoro-82M (TTS)
- **Backup ASR finding:** confirmed via direct testing that MMS (1,162 supported languages) does not cover Balti (`bft`) at all — falls back to Tibetan (`bod`), which produces wrong-script, low-quality output on real Balti audio. Rather than silently pass this downstream, the pipeline flags fallback output as unreliable and skips translation, returning a clear warning instead of a confidently wrong result.
- Verified both paths directly:
  - **Happy path:** real test clip → Wav2Vec2 transcript closely matched ground truth → NLLB produced coherent English translation → Kokoro produced working audio output
  - **Failover path:** Primary ASR forced to fail → MMS fallback correctly triggered → unreliable flag correctly prevented bad output from reaching translation
- Migrated from Lightning.ai to Kaggle notebooks mid-project after Lightning credits ran low; pipeline built and verified entirely on Kaggle's free GPU tier
- Code pushed to GitHub as `pipeline.py`

## Window (not started)
Demo deployment — Gradio, public link.

## Flow (not started)
Chunked streaming pipeline, latency measurement.

## Polish (not started)
Final writeup, docs, Hurdles Log, video/showcase decisions.
