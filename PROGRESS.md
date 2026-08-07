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
Fine-tuned primary ASR (Whisper-small) on the Balti dataset.
- Manual feature-extraction/tokenization pipeline built (`process_split`), corrupted-audio filtering added (`soundfile` readability check)
- Hit and fixed: processed data never wired into the Trainer — `Seq2SeqTrainer` was still pointed at the raw, unprocessed dataset instead of the processed one
- Hit and fixed: kernel OOM crash converting the full processed dataset to a `Dataset` via `from_list` (~8k examples × full `input_features` arrays exceeded RAM) — switched to `Dataset.from_generator` to stream to disk instead of materializing in memory
- Clean training curve, no overfitting: WER dropped 63.42% → 36.73% over 1000 steps; best checkpoint at step 900 (val loss 0.304, WER 36.73%)
- Pushed to `YuvrajGujari/whisper-small-balti`; local checkpoint-900 weights cross-checked against the pushed weights via the Hugging Face webpage (model card WER display initially looked like it reflected step 1000, not the best checkpoint — verification done to confirm which weights actually shipped)
- Gap vs. published BaltiVoice benchmark (26.74% WER on a similar corpus) noted for the writeup — worth investigating rather than treating current result as ceiling
- Code changes (Trainer wiring fix, OOM fix) committed and pushed to GitHub

## Echo (not started)
Fine-tune backup ASR (wav2vec2-XLS-R-300M).

## Bridge ✅
Fine-tuned `facebook/nllb-200-distilled-600M` on the `bft_Arab` BOUQuET data (504 train, 854 eval pairs) — the project's real novel contribution, as no existing MT solution for Balti exists.
- Added `bft_Arab` as a new special token, embedding warm-started from `bod_Tibt`
- Best checkpoint: step 650, BLEU 4.883 — val loss climbed steadily past step 300 (overfitting, same pattern seen in Whisper Round 2), reported honestly as the real ceiling this data size allows
- Verified via SHA256, pushed to `YuvrajGujari/nllb-balti-mt`; disk cleaned up (non-best checkpoints removed)

## Pipeline Assembly (VAD → ASR → MT → TTS) ✅
- Assembled full end-to-end pipeline: Silero VAD → fine-tuned Whisper-small-balti (primary ASR) → MMS-1b-all (pretrained backup ASR) → fine-tuned NLLB-balti-mt (translation) → Kokoro-82M (TTS)
- **Backup ASR finding:** confirmed via direct testing that MMS (1,162 supported languages) does not cover Balti (`bft`) at all — falls back to Tibetan (`bod`), which produces wrong-script, low-quality output on real Balti audio. Rather than silently pass this downstream, the pipeline flags fallback output as unreliable and skips translation, returning a clear warning instead of a confidently wrong result.
- Verified both paths directly:
  - **Happy path:** real test clip → Whisper transcript closely matched ground truth → NLLB produced coherent English translation → Kokoro produced working audio output
  - **Failover path:** Whisper forced to fail → MMS fallback correctly triggered → unreliable flag correctly prevented bad output from reaching translation
- Migrated from Lightning.ai to Kaggle notebooks mid-project after Lightning credits ran low; pipeline built and verified entirely on Kaggle's free GPU tier
- Code pushed to GitHub as `pipeline.py` (dropped phase numbering going forward — the project stopped mapping cleanly to a fixed phase sequence after the language pivot)

## Window (not started)
Demo deployment — Gradio, public link.

## Flow (not started)
Chunked streaming pipeline, latency measurement.

## Polish (not started)
Final writeup, docs, Hurdles Log, video/showcase decisions.
