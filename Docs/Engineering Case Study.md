# Balti Tarjuman – Engineering Case Study

> *Building an end-to-end Balti → English speech translation system for one of the world's lowest-resource languages.*

---

# Introduction

Balti Tarjuman was built to explore the challenges of developing an end-to-end speech translation system for **Balti**, a severely low-resource language with very limited publicly available datasets, pretrained models, and language resources.

Unlike high-resource languages where complete solutions already exist, almost every stage of this project required investigation, validation, adaptation, or redesign before it could become part of the final pipeline.

This document records the major engineering challenges encountered during development, the decisions taken to overcome them, and the lessons learned throughout the project.

---

# System Overview

The pipeline consists of the following stages, available in both a **batch** mode and a **streaming, real-time** mode:

```text
Balti Speech
      │
      ▼
Voice Activity Detection (VAD)
      │
      ▼
Automatic Speech Recognition (ASR)
      │
      ▼
Balti Text
      │
      ▼
Machine Translation (MT)
      │
      ▼
English Text
      │
      ▼
Text-to-Speech (TTS)
      │
      ▼
English Speech
```

Each stage was developed independently before being integrated into the complete pipeline. The batch pipeline (`pipeline.py`) exposes this as a class-based API, and the streaming pipeline (`streaming_pipeline.py`) builds directly on top of it — importing the same class rather than duplicating logic — to deliver continuous, near real-time translation from a live microphone feed.

---

# Engineering Challenges

## Challenge 1 — Understanding the Balti AI Ecosystem

### Problem

Before writing any code, I needed to understand what resources already existed for Balti.

Unlike widely supported languages, there was no obvious collection of datasets, pretrained models, benchmarks, or translation systems that could simply be assembled into a working pipeline.

### Approach

I surveyed publicly available datasets, Hugging Face models, multilingual foundation models, and recent publications related to Balti.

This investigation helped identify which parts of the pipeline already had usable resources and which parts would need to be built or adapted.

### Outcome

The project architecture was designed around the resources that actually existed rather than assumptions.

### Lesson Learned

Understanding the ecosystem before implementation prevents costly redesigns later in the project.

---

## Challenge 2 — Finding a Reliable Translation Dataset

### Problem

The first Balti translation model I evaluated appeared promising but failed during validation.

Although its name suggested Balti translation support, it produced text in the wrong script, used the wrong translation direction, and lacked proper language-code infrastructure.

### Approach

Instead of forcing the model into the pipeline, I continued evaluating available resources until I discovered Facebook's **Bouquet** dataset containing the `bft_Arab` configuration.

Unlike previous candidates, it matched the writing system used throughout the project.

### Outcome

The translation stage was rebuilt using a dataset that aligned with the rest of the pipeline.

### Lesson Learned

Model names can be misleading. Every model should be validated before becoming part of a production pipeline.

---

## Challenge 3 — Adapting NLLB for Balti

### Problem

The original translation stage was planned around NLLB-200.

After inspecting the tokenizer configuration directly, I confirmed that Balti was not natively supported.

Without a valid language token, the planned fine-tuning strategy could not work.

### Approach

Instead of abandoning the architecture, I investigated tokenizer internals, language codes, and multilingual datasets.

A new `bft_Arab` language token was introduced and its embedding initialized from the closest available language representation before fine-tuning.

### Outcome

The existing architecture could be preserved while extending the model to support Balti.

### Lesson Learned

Documentation should never replace verification. Inspecting model internals often reveals limitations that are not immediately obvious.

---

## Challenge 4 — Building a Reliable Dataset Download Pipeline

### Problem

Downloading thousands of speech files exposed multiple operational problems including HTTP rate limits, interrupted downloads, stale cache files, and corrupted partial downloads.

These failures made the dataset pipeline unreliable.

### Approach

The download process was redesigned using resumable downloads, retry logic, exponential backoff, cache cleanup, and `snapshot_download`.

The goal shifted from downloading quickly to downloading reliably.

### Outcome

Dataset acquisition became reproducible and resilient against transient network failures.

### Lesson Learned

Reliable data acquisition is just as important as model training.

---

## Challenge 5 — Cleaning the Speech Dataset

### Problem

The downloaded corpus contained inconsistencies between metadata and audio files.

Some recordings had no matching metadata while some metadata entries pointed to missing audio.

### Approach

Rather than relying on strict automated loading, the dataset was rebuilt manually using a verified inner join between metadata and available audio files.

Only valid speech–transcript pairs were retained.

### Outcome

The final training dataset contained only verified samples suitable for model training.

### Lesson Learned

Data quality problems should be solved before training rather than compensated for afterward.

---

## Challenge 6 — Whisper Fine-Tuning

### Problem

The ASR pipeline encountered several engineering issues during preprocessing including dependency conflicts, audio decoding failures, repeated kernel restarts, incorrect dataset wiring, and memory limitations.

### Approach

Instead of continuing to patch individual issues, the preprocessing pipeline was simplified.

Audio decoding was performed directly from embedded bytes, processed datasets were rebuilt correctly, and generator-based dataset creation replaced memory-intensive approaches.

### Outcome

Training became significantly more stable and reproducible.

### Lesson Learned

Simpler pipelines are often more reliable than complex dependency chains.

---

## Challenge 7 — Training a Backup ASR Model

### Problem

Relying entirely on a single speech recognition model introduced unnecessary project risk.

### Approach

A second ASR model based on wav2vec2 was trained in parallel.

The complete processor, tokenizer, and configuration were preserved alongside the model weights to ensure reproducible inference.

### Outcome

The project gained an independent backup ASR path while also enabling architectural comparison.

### Lesson Learned

Building redundancy into critical components makes long-term projects more robust.

---

## Challenge 8 — Platform Migration

### Problem

Cloud compute availability changed during development, requiring the project to move to a different platform before experimentation had finished.

### Approach

The development environment was recreated from scratch, including package installation, authentication, Git configuration, secrets management, and project structure.

The workflow was redesigned to remain portable rather than tied to a single provider.

### Outcome

Development continued with minimal disruption.

### Lesson Learned

Machine learning projects should be portable enough to survive infrastructure changes.

---

## Challenge 9 — Building a Safe End-to-End Pipeline

### Problem

Large multilingual speech models advertised broad language coverage, but direct validation showed that Balti was not always supported.

Passing unsupported predictions into later pipeline stages could silently produce incorrect translations.

### Approach

Instead of allowing uncertain predictions to propagate, explicit validation and fallback handling were introduced so unsupported outputs are detected and reported.

### Outcome

The pipeline now prioritizes reliability over producing misleading results.

### Lesson Learned

A system that clearly reports uncertainty is more trustworthy than one that silently produces incorrect output.

---

## Challenge 10 — Packaging the Pipeline as a Reusable Class

### Problem

The working pipeline existed as a set of scripts rather than a coherent, reusable component. This made it harder to reason about, harder to reuse across the batch and streaming variants, and easy to leave stale references behind as pieces changed — including a leftover backup-ASR reference that still pointed at a generic pretrained model instead of the project's own fine-tuned one.

### Approach

The pipeline was refactored into a class-based `BaltiTarjumanPipeline`, consolidating VAD, ASR, MT, and TTS behind a single, testable interface. Refactoring surfaced the stale backup-ASR reference, which was corrected to point at the project's fine-tuned model.

### Outcome

A single, reusable pipeline class that both the batch and streaming entry points build on, with no duplicated model-loading or inference logic between them.

### Lesson Learned

Packaging code into a proper interface isn't just cleanup — the act of refactoring surfaces stale assumptions that are easy to miss when logic is scattered across scripts.

---

## Challenge 11 — Real-Time Streaming Translation

### Problem

The batch pipeline processed complete audio files, but a real-time speech translator needs to process a continuous, unsegmented audio stream — detecting when someone starts and stops speaking, and feeding each spoken segment through the pipeline with low enough latency to feel conversational.

### Approach

A `VADSegmenter`, built on Silero's `VADIterator`, was combined with a threaded `StreamingBaltiTarjumanPipeline` that used worker threads and queues to keep audio capture, inference, and playback from blocking one another. Getting this working end-to-end surfaced four distinct bugs, each traced to a clear root cause:

1. **Cross-file state leakage** — the VAD segmenter's internal "triggered" state carried over between separate file tests, since it was never reset. Fixed by automatically resetting the segmenter's state per file.
2. **Segments that never closed** — a segment ending right at the minimum-silence threshold could get stuck with no "end" event ever firing, silently dropping the tail of an utterance. Fixed by adding an explicit `flush()` method to force-close a pending segment.
3. **Truncated TTS output** — Kokoro's TTS yields multiple audio chunks for longer sentences, but the code only kept the first one, silently truncating anything past a short phrase. Fixed by concatenating all yielded chunks before returning audio.
4. **Interrupted live playback** — Gradio's live output component kept cutting off mid-playback because the streaming callback returned `None` on every idle tick, which Gradio interprets as "clear the player." Fixed by returning `gr.skip()` on idle ticks instead of `None`.

Latency was further reduced by loading models in fp16 and keeping TTS entirely in-memory, avoiding disk writes/reads in the streaming path.

### Outcome

Confirmed working end-to-end via live microphone input through Gradio, with full sentences intact and round-trip latency of roughly 2–3 seconds — within the project's 2–4 second target for conversational feel.

### Lesson Learned

Streaming systems fail in ways batch systems don't — state that leaks across boundaries, events that never fire, output that gets silently truncated, and UI frameworks with implicit conventions (like `None` meaning "clear") are all invisible until you test the continuous, real-time path specifically. Each of these four bugs would have been undetectable from single-file batch testing alone.

---

# Key Engineering Takeaways

Throughout the project, several principles consistently proved valuable:

* Validate assumptions before building around them.
* Inspect models instead of relying only on documentation.
* Treat data quality as a first-class engineering problem.
* Build systems that can recover from operational failures.
* Prefer simplifying pipelines over adding more complexity.
* Design infrastructure that is portable across platforms.
* Fail safely instead of silently.
* Test the continuous, real-time path explicitly — batch testing alone won't surface streaming-specific failure modes.

---

# Future Improvements

Current areas for future work include:

* Longer Whisper fine-tuning with matched training budgets.
* Expanded Balti–English parallel data, to move MT quality beyond its current small-scale training set.
* More comprehensive evaluation across multiple ASR models.
* Automated experiment tracking.
* Containerized deployment for reproducibility.

---

# Closing Thoughts

Balti Tarjuman has been an exercise in far more than model training. Building an end-to-end system for a low-resource language required solving challenges across data engineering, multilingual NLP, speech processing, cloud infrastructure, system integration, and — in its final phase — real-time systems design.

The most valuable lesson from this project is that successful machine learning systems are built by consistently solving many interconnected engineering problems — not by relying on a single model or algorithm. The streaming phase in particular reinforced this: a pipeline that works correctly on batch input is not the same thing as a pipeline that works correctly in real time, and the gap between the two is where the more interesting engineering problems live.
