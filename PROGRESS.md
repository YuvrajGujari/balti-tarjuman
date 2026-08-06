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

## Voice 🔄 (in progress)
Fine-tuning primary ASR (Whisper-small) on the Balti dataset.
- Manual feature-extraction/tokenization pipeline built (`process_split`), corrupted-audio filtering added (`soundfile` readability check)
- Hit and fixed: processed data never wired into the Trainer — `Seq2SeqTrainer` was still pointed at the raw, unprocessed dataset instead of the processed one
- Hit and fixed: kernel OOM crash converting the full processed dataset to a `Dataset` via `from_list` (~8k examples × full `input_features` arrays exceeded RAM) — switched to `Dataset.from_generator` to stream to disk instead of materializing in memory
- Training run not yet completed

## Echo (not started)
Fine-tune backup ASR (wav2vec2-XLS-R-300M).

## Bridge ✅
Fine-tuned `facebook/nllb-200-distilled-600M` on the `bft_Arab` BOUQuET data (504 train, 854 eval pairs) — the project's real novel contribution, as no existing MT solution for Balti exists.
- Added `bft_Arab` as a new special token, embedding warm-started from `bod_Tibt`
- Best checkpoint: step 650, BLEU 4.883 — val loss climbed steadily past step 300 (overfitting, same pattern seen in Whisper Round 2), reported honestly as the real ceiling this data size allows
- Verified via SHA256, pushed to `YuvrajGujari/nllb-balti-mt`; disk cleaned up (non-best checkpoints removed)

## Assembly (not started)
Full VAD → ASR (+ fallback) → MT → TTS pipeline, batch mode.

## Window (not started)
Demo deployment — Gradio, public link.

## Flow (not started)
Chunked streaming pipeline, latency measurement.

## Polish (not started)
Final writeup, docs, Hurdles Log, video/showcase decisions.
