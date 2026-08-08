# 🏔️ Balti Tarjuman (بلتی ترجمان)

[![Hugging Face Champion Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-whisper--small--balti-yellow)](https://huggingface.co/YuvrajGujari/whisper-small-balti)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen)](https://www.python.org/)

An end-to-end Speech-to-Speech Translation (S2ST) system for **Balti (بلتی)**, a low-resource Tibetic language spoken by ~400,000 people across Gilgit-Baltistan and Baltistan, written in Perso-Arabic (Nastaliq) script. 

Historically, Balti has had virtually no representation in natural language processing (NLP) or speech technology—lacking dedicated ASR baselines, translation engines, or coverage in major multilingual benchmarks. **Balti Tarjuman** bridges this gap by deploying an integrated pipeline that translates spoken Balti into synthesized English speech.

---

## 🏗️ System Architecture

The pipeline processes continuous audio through four sequential stages:
┌─────────────────┐     ┌──────────────────────┐     ┌───────────────────────┐     ┌────────────────┐
│   Silero VAD    │ ──> │ Fine-tuned Whisper   │ ──> │   Fine-tuned NLLB    │ ──> │  Kokoro-82M    │
│ (Voice Activity)│     │  (ASR Engine)        │     │  (Translation Engine) │     │  (TTS Engine)  │
└─────────────────┘     └──────────────────────┘     └───────────────────────┘     └────────────────┘

1. **Voice Activity Detection (VAD):** [Silero VAD](https://github.com/snakers4/silero-vad) isolates valid speech frames and trims silent segments.
2. **Automatic Speech Recognition (ASR):** Fine-tuned [`openai/whisper-small`](https://huggingface.co/YuvrajGujari/whisper-small-balti) transcribes Balti audio into Perso-Arabic text.
3. **Machine Translation (MT):** Fine-tuned [`facebook/nllb-200-distilled-600M`](https://huggingface.co/YuvrajGujari/nllb-balti-mt) translates Balti text into English.
4. **Text-to-Speech (TTS):** [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) synthesizes English translations into audio.

---

## 🏆 ASR Benchmark Leaderboard

By applying **Cold-Start re-initialization** and **SpecAugment regularization** (`mask_time_prob=0.05`, `mask_feature_prob=0.05`) with a high learning rate ($1\times 10^{-4}$), our fine-tuned Whisper model established a new State-of-the-Art performance record for Balti ASR:

| Rank | Model Architecture | Strategy | Steps | Validation WER | Link |
| :---: | :--- | :--- | :---: | :---: | :---: |
| 🥇 **1** | **Whisper-small (Champion)** | **Cold-Start + SpecAugment** | **2500** | **17.40%** | [HF Model](https://huggingface.co/YuvrajGujari/whisper-small-balti) |
| 🥈 **2** | Wav2Vec2 XLS-R 300M | Cold-Start CTC | 7560 | **22.82%** | [HF Model](https://huggingface.co/YuvrajGujari/wav2vec2-balti) |
| 🥉 **3** | *BaltiVoice Paper Baseline* | Published Literature | — | *26.74%* | — |
| 4 | Whisper-small (Warm-Start R2) | Standard Fine-Tuning (`1e-5` LR) | 1000 | **36.38%** | — |
| 5 | Whisper-small (Zero-Shot) | Base Out-of-the-Box | 0 | **63.42%** | — |

---

## 🔗 Models & Datasets

| Component | Repository Link | Description |
| :--- | :--- | :--- |
| 🏆 **ASR Champion** | [`YuvrajGujari/whisper-small-balti`](https://huggingface.co/YuvrajGujari/whisper-small-balti) | Fine-tuned Whisper-small (**17.40% WER**) |
| ⚡ **ASR Baseline** | [`YuvrajGujari/wav2vec2-balti`](https://huggingface.co/YuvrajGujari/wav2vec2-balti) | Fine-tuned Wav2Vec2 XLS-R 300M (**22.82% WER**) |
| 🌐 **MT Model** | [`YuvrajGujari/nllb-balti-mt`](https://huggingface.co/YuvrajGujari/nllb-balti-mt) | Fine-tuned NLLB-200 Distilled (**4.88 BLEU**) |
| 📂 **ASR Dataset** | [`YuvrajGujari/balti-tarjuman-data`](https://huggingface.co/datasets/YuvrajGujari/balti-tarjuman-data) | Cleaned Balti audio with Perso-Arabic text |

---

## ⚡ Quick Start

### Installation

```bash
git clone [https://github.com/YuvrajGujari/balti-tarjuman.git](https://github.com/YuvrajGujari/balti-tarjuman.git)
cd balti-tarjuman
pip install -r requirements.txt
#End-to-End Speech Translation
import torch
import soundfile as sf
from pipeline import BaltiTarjumanPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = BaltiTarjumanPipeline(device=device)

# Translate Balti speech file to English audio
translated_text, synthesized_audio = pipeline.translate("path/to/balti_audio.wav")

print("English Translation:", translated_text)
sf.write("output_english.wav", synthesized_audio, samplerate=24000)
