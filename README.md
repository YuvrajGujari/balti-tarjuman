# Balti Tarjuman

A speech-to-speech translation system for Balti, a Tibetic language spoken by an estimated ~400,000 people in the Gilgit-Baltistan and Baltistan region, written in a Nastaliq-based script. Balti has had almost no presence in NLP or speech technology — no prior ASR system, no MT system, and no coverage in major multilingual benchmarks until very recently.

This project pivoted from an earlier Pashto-language version of the same architecture (kept private due to an unrelated client-project conflict), retargeting the full pipeline to Balti.

## Pipeline
Speech (Balti) → VAD (Silero) → ASR (fine-tuned Whisper-small) → MT (fine-tuned NLLB-200) → TTS (Kokoro-82M) → Speech (English)

## Status
- MT stage: fine-tuned, BLEU 4.88 (limited by small parallel-text availability — see PROGRESS.md)
- ASR stage: in progress
