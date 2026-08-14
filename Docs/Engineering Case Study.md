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

Each stage was developed independently before being integrated into the complete pipeline. The batch pipeline (`pipeline.py`) exposes this as a class-based API, and the streaming pipeline (`streaming_pipeline.py`) builds directly on top of it — importing the same class rather than duplicating logic — to deliver continuous, near real-time translation from a live microphone feed. The ASR stage itself is resilient by design: a primary model with a bounded-latency fallback to a backup model, detailed in Challenge 12 below.

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

## Challenge 12 — Making the ASR Fallback Actually Latency-Aware (and Testing It)

### Problem

The original ASR fallback (Challenge 9) only triggered on a Whisper *exception* — it had no defense against Whisper simply running slow on a given input. For the live streaming path specifically, this mattered: a single unusually slow inference could stall the whole session, with no mechanism to detect or route around it. The fallback also had zero deliberate test coverage — it had never actually been forced to trigger and observed end-to-end, only reasoned about.

### Approach

Implemented a bounded-latency fallback: Whisper now runs inside a worker thread via `concurrent.futures.ThreadPoolExecutor`, and if it doesn't return within a configurable `whisper_timeout_sec`, the pipeline stops waiting and falls through to the wav2vec2 backup immediately, tagging the result as `wav2vec2_fallback_timeout` (distinct from `wav2vec2_fallback_error` for the original exception path — useful for knowing which failure mode is actually occurring in practice).

To set the timeout threshold from real data rather than a guess, I built a small test script that pulls held-out clips directly from the project's own test split (`YuvrajGujari/balti-tarjuman-data`) and measures Whisper's actual wall-clock latency. Two separate measurement runs (n=20 each) gave median latency around 0.70–0.72s but noticeably different p95/max values (3.01s vs. 1.26s) — likely GPU warm-up effects or natural spread across a small sample, not a real change in the model. Rather than picking whichever number looked better, I treated the variance itself as the finding: median latency is stable, but tail latency needs a conservative margin, not a single-run point estimate.

The same test script also deliberately forced the timeout path (`whisper_timeout_sec=0.01`, an impossible threshold) on a real clip and asserted that the pipeline correctly routed to `wav2vec2_fallback_timeout` and returned a valid, non-empty transcript — not just that the code ran without raising an error.

### Outcome

This deliberate test immediately surfaced a real, previously-undetected bug: the wav2vec2 model is loaded in fp16 on CUDA, but the processor's output stayed in the default float32, causing a `RuntimeError: Input type (float) and bias type (c10::Half) should be the same` the moment the fallback path actually ran. This bug had been present since the original exception-only fallback — it simply never triggered during normal testing, since Whisper rarely raised exceptions on clean test clips. Fixed by explicitly casting float tensors to match the model's dtype before the forward pass.

With the bug fixed, both fallback paths (timeout and exception) were confirmed working end-to-end: correct routing, correct tagging, valid transcript returned. The streaming pipeline was also updated to track a running per-session count of which ASR path served each segment (`whisper` / `wav2vec2_fallback_timeout` / `wav2vec2_fallback_error`), so a live session's actual fallback trigger rate can be reported as a real operational statistic rather than something only validated in isolated testing.

One design tradeoff is stated honestly rather than glossed over: this is a thread-level timeout, not true CUDA-level cancellation. An abandoned Whisper call keeps running on the GPU in the background until it finishes, competing for compute with the wav2vec2 fallback that starts immediately after — a timeout event is therefore momentarily *more* GPU-expensive, not a free escape hatch.

### Lesson Learned

A fallback path that has never been deliberately, forcibly triggered and observed is not a tested fallback — it's an assumption. The dtype bug here had nothing to do with the timeout logic itself; it was a latent, unrelated bug sitting in code that looked correct and had simply never executed under real conditions. This is the same category of lesson as the streaming bugs in Challenge 11: reliability mechanisms need to be exercised under the exact failure condition they exist to handle, not just reasoned about from the happy path.

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
* A fallback or safety mechanism that has never been deliberately forced to trigger is unverified, not tested — the failure path itself can silently harbor bugs.
* When a measurement varies across repeated runs, report the variance honestly and design conservatively around it, rather than presenting a single favorable run as the definitive number.

---

# Future Improvements

Current areas for future work include:

* Longer Whisper fine-tuning with matched training budgets.
* Expanded Balti–English parallel data, to move MT quality beyond its current small-scale training set.
* More comprehensive evaluation across multiple ASR models.
* Automated experiment tracking.
* Containerized deployment for reproducibility.
* A true CUDA-level cancellation for the ASR timeout fallback, so an abandoned Whisper call stops consuming GPU resources immediately rather than finishing in the background.
* A larger-scale, statistically robust latency benchmark (more than n=20, multiple hardware configurations) to replace the current informally-measured timeout threshold with a rigorously validated one.

---

# Closing Thoughts

Balti Tarjuman has been an exercise in far more than model training. Building an end-to-end system for a low-resource language required solving challenges across data engineering, multilingual NLP, speech processing, cloud infrastructure, system integration, and — in its final phases — real-time systems design and reliability engineering.

The most valuable lesson from this project is that successful machine learning systems are built by consistently solving many interconnected engineering problems — not by relying on a single model or algorithm. The streaming phase reinforced that a pipeline correct on batch input is not the same thing as one correct in real time; the ASR fallback work reinforced a related point one level deeper — a reliability mechanism that has never been deliberately tested under its own failure condition isn't actually reliable, it's untested. Both lessons point the same direction: the gap between "should work" and "verified to work" is where the real engineering happens.
