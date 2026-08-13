# 🏔️ Balti Tarjuman (بلتی ترجمان)

[![Hugging Face Champion Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-whisper--small--balti-yellow)](https://huggingface.co/YuvrajGujari/whisper-small-balti)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen)](https://www.python.org/)

An end-to-end Speech-to-Speech Translation (S2ST) system for **Balti (بلتی)**, a low-resource Tibetic language spoken by ~400,000 people across Gilgit-Baltistan and Baltistan, written in Perso-Arabic (Nastaliq) script.

Historically, Balti has had virtually no representation in natural language processing (NLP) or speech technology—lacking dedicated ASR baselines, translation engines, or coverage in major multilingual benchmarks. **Balti Tarjuman** bridges this gap by deploying an integrated pipeline that translates spoken Balti into synthesized English speech — available both as a **batch pipeline** and as a **real-time streaming pipeline** (~2–3s round-trip latency from live microphone input).

---

## 🎥 Demo

**Upload a Balti audio file and hear the English translation:**

![Balti Tarjuman upload demo](https://github.com/user-attachments/assets/d3b61376-5174-4c32-a406-26427b6565a2)

**Live, unscripted — speaking Balti into the mic:**

![Balti Tarjuman live mic demo](https://github.com/user-attachments/assets/e7048524-c1e4-4a7e-b19e-8ef65dec22f1)

🎬 [Watch the full demo with audio (1:04)](https://github.com/user-attachments/assets/ffed7c0f-b8a0-4312-b6d5-eccdb7e58e85)

---

Here is a updated version of your **System Architecture** section, formatted in clean Markdown ready to copy directly into your `README.md`.

It incorporates the **fallback router**, the **Mermaid diagram with decision logic**, and explicit mentions of your **resilience & failover engineering**.

---

### Markdown to Copy/Paste into `README.md`

```markdown
## 🏗️ System Architecture & Failover Design

The pipeline processes continuous audio through four sequential stages with integrated **circuit-breaker resilience**:

```mermaid
flowchart LR
    A[🎙️ Silero VAD<br/>Voice Activity Detection] --> B{ASR Router}
    B -- Primary Engine<br/>17.40% WER --> C[📝 Fine-tuned Whisper-small]
    B -- Latency/Timeout Fallback<br/>22.11% WER --> D[⚡ Fine-tuned Wav2Vec2 XLS-R]
    C --> E[🌐 Fine-tuned NLLB-200<br/>Translation Engine]
    D --> E
    E --> F[🔊 Kokoro-82M<br/>TTS Engine]

```

1. **Voice Activity Detection (VAD):** [Silero VAD](https://github.com/snakers4/silero-vad) isolates valid speech frames, drops silent segments, and handles dynamic noise floor trimming.
2. **Resilient ASR Engine (Whisper + Wav2Vec2 Fallback):**
* **Primary Engine:** Fine-tuned [`openai/whisper-small`](https://huggingface.co/YuvrajGujari/whisper-small-balti) (**17.40% WER**) transcribes Balti speech into Perso-Arabic (Nastaliq) text.
* **Fallback Circuit Breaker:** If Whisper exceeds processing latency bounds or encounters decoder stalls on noisy live inputs, the system gracefully route-fails to an encoder-only CTC model—fine-tuned [`wav2vec2-xls-r-300m`](https://huggingface.co/YuvrajGujari/wav2vec2-balti-specaugment) (**22.11% WER**). This non-autoregressive pass guarantees predictable execution times and prevents dropped frames during live streaming.


3. **Machine Translation (MT):** Fine-tuned [`facebook/nllb-200-distilled-600M`](https://huggingface.co/YuvrajGujari/nllb-balti-mt) converts translated Perso-Arabic Balti text into target English syntax.
4. **Text-to-Speech (TTS):** [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) synthesizes high-quality English audio from the translated text.

---

### Pipeline Execution Modes

The unified pipeline operates in two modes sharing a single model lifecycle:

* **Batch Mode** (`pipeline.py`) — Processes standard `.wav` files via `BaltiTarjumanPipeline.run()` for full offline translations.
* **Streaming Mode** (`streaming_pipeline.py`) — Real-time audio streaming built on top of Silero's `VADIterator` coupled with an asynchronous, multi-threaded worker/queue pipeline. Captures, infers, and outputs translated speech with low round-trip latency (~2–3s) without locking system audio buffers or duplicating GPU memory overhead.

---

## 🏆 ASR Benchmark Leaderboard

By applying **Cold-Start re-initialization** and **SpecAugment regularization** (`mask_time_prob=0.05`, `mask_feature_prob=0.05`) with a high learning rate ($1\times 10^{-4}$), our fine-tuned Whisper model established a new state-of-the-art result for Balti ASR:

| Rank | Model Architecture | Strategy | Steps | Validation WER | Link |
| :---: | :--- | :--- | :---: | :---: | :---: |
| 🥇 **1** | **Whisper-small (Champion)** | **Cold-Start + SpecAugment** | **2500** | **17.40%** | [HF Model](https://huggingface.co/YuvrajGujari/whisper-small-balti) |
| 🥈 **2** | Wav2Vec2 XLS-R 300M | SpecAugment + Tuned LR | 7560 | **22.11%** | [HF Model](https://huggingface.co/YuvrajGujari/wav2vec2-balti-specaugment) |
| 🥉 **3** | *BaltiVoice Paper Baseline* | Published Literature | — | *26.74%* | — |
| 4 | Wav2Vec2 XLS-R 300M | Cold-Start CTC (no augmentation) | 7400 | **22.82%** | [HF Model](https://huggingface.co/YuvrajGujari/wav2vec2-balti) |
| 5 | Whisper-small (Warm-Start R2) | Standard Fine-Tuning (`1e-5` LR) | 1000 | **36.38%** | — |
| 6 | Whisper-small (Zero-Shot) | Base Out-of-the-Box | 0 | **63.42%** | — |

---

## 🔗 Models & Datasets

| Component | Repository Link | Description |
| :--- | :--- | :--- |
| 🏆 **ASR Champion** | [`YuvrajGujari/whisper-small-balti`](https://huggingface.co/YuvrajGujari/whisper-small-balti) | Fine-tuned Whisper-small (**17.40% WER**) |
| ⚡ **ASR Backup** | [`YuvrajGujari/wav2vec2-balti-specaugment`](https://huggingface.co/YuvrajGujari/wav2vec2-balti-specaugment) | Fine-tuned Wav2Vec2 XLS-R 300M, SpecAugment + tuned LR (**22.11% WER**) |
| 🌐 **MT Model** | [`YuvrajGujari/nllb-balti-mt`](https://huggingface.co/YuvrajGujari/nllb-balti-mt) | Fine-tuned NLLB-200 Distilled (**4.88 BLEU**) |
| 📂 **ASR Dataset** | [`YuvrajGujari/balti-tarjuman-data`](https://huggingface.co/datasets/YuvrajGujari/balti-tarjuman-data) | Cleaned Balti audio with Perso-Arabic text |

---

## ⚡ Quick Start

### Installation

```bash
git clone https://github.com/YuvrajGujari/balti-tarjuman.git
cd balti-tarjuman
pip install -r requirements.txt
```

### End-to-End Speech Translation (Batch)

```python
from pipeline import BaltiTarjumanPipeline

pipeline = BaltiTarjumanPipeline()  # auto-selects CUDA if available

result = pipeline.run("path/to/balti_audio.wav")

print("Balti transcript:", result["balti_text"])
print("English translation:", result["english_text"])
print("Output audio written to:", result["audio_path"])
```

### Real-Time Streaming Translation

```python
from pipeline import BaltiTarjumanPipeline
from streaming_pipeline import StreamingBaltiTarjumanPipeline

pipeline = BaltiTarjumanPipeline()
streaming = StreamingBaltiTarjumanPipeline(pipeline)  # reuses pipeline's loaded models
streaming.start()

# Feed a complete wav file (or live mic frames via streaming.feed_audio_frame())
streaming.feed_wav_file("path/to/balti_audio.wav")

result = streaming.get_next_output(timeout=5)
if result:
    print("English translation:", result["english_text"])
```

---

## 📖 Engineering Case Study

For a detailed account of the challenges solved along the way — dataset acquisition, adapting NLLB for an unsupported language, ASR fine-tuning strategy, and building the real-time streaming layer — see [`docs/Engineering Case Study.md`](docs/Engineering%20Case%20Study.md).

---

## 📄 License

Apache 2.0 — see [LICENSE](LICENSE) for details.
