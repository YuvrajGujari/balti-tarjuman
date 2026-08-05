# Balti Tarjuman — Progress Log

Balti-to-English speech-to-speech translation — a personal project building a full pipeline (speech recognition, translation, and speech synthesis) for Balti, a low-resource language with very limited existing digital tooling.

---

## Foundation ✅
- Cloud dev environment set up: Lightning.ai Studio, VS Code Remote-SSH, GitHub repo, Hugging Face account
- Secrets handled via `.env` + `python-dotenv` from the start
- VS Code Auto Save enabled from the start to avoid unsaved-file issues

## First Listen ✅
Zero-shot `whisper-small` on Balti audio: complete failure, as expected for a language the model has no exposure to. Output scattered across unrelated scripts/languages (Khmer, French-like, Armenian-like, Hebrew-like, Chinese characters) — no coherent signal at all. Confirms fine-tuning is necessary, not just helpful.

## Gathering 🔄 (in progress)
- **ASR dataset:** `NasuAhmed/balti-voice-dataset` — 9,481 raw rows; hit permission and timeout errors downloading, resolved with a retry loop (exponential backoff) around `snapshot_download`; found and cleaned a metadata/audio mismatch (some files had no metadata row and vice versa) via inner join
- **MT dataset:** resolved after real investigation — NLLB-200 has zero native Balti (`bft`) coverage; found `facebook/bouquet`'s `bft_Arab` config (correct script, English-paired, curated) — small (504 dev + 854 test = 1,358 pairs) but usable and honest to report as-is
- Prior published work exists: "BaltiVoice" paper (arXiv 2606.03504), 26.74% WER Whisper-small ASR — our differentiator is the **full pipeline** (ASR+MT+TTS), not ASR alone

## Voice (not started)
Fine-tune primary ASR (Whisper-small) on the Balti dataset.

## Echo (not started)
Fine-tune backup ASR (wav2vec2-XLS-R-300M).

## Bridge (not started)
Fine-tune `facebook/nllb-200-distilled-600M` on the `bft_Arab` BOUQuET data. This is the project's real novel contribution — no existing MT solution for Balti exists. Small dataset means a modest step count and close overfitting watch; will honestly report the real ceiling this data size allows.

## Assembly (not started)
Full VAD → ASR (+ fallback) → MT → TTS pipeline, batch mode.

## Window (not started)
Demo deployment — Gradio, public link.

## Flow (not started)
Chunked streaming pipeline, latency measurement.

## Polish (not started)
Final writeup, docs, Hurdles Log, video/showcase decisions.
