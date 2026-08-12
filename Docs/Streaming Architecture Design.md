# Streaming Architecture Design — Balti Tarjuman

**Target:** 2-4s end-to-end latency (speech spoken → translated speech heard), chunked near-real-time — not sub-second token-level streaming.

---

## Why not sub-second

True sub-second streaming needs token-level streaming ASR (partial hypotheses as audio arrives) and streaming TTS (audio synthesized before the full sentence is known). Whisper and Kokoro don't natively support that here. The 2-4s target is achieved instead by processing short, complete speech segments as fast as possible — segment-level pipelining, not token-level streaming. This matches the project's existing target.

---

## Pipeline shape

```
Mic input (raw audio frames)
      |
      v
[1] Streaming VAD Segmenter  --(speech segment, ~1-6s each)-->  queue
      |
      v
[2] ASR (Whisper, wav2vec2 fallback)  --(Balti text)-->
      |
      v
[3] MT (NLLB)  --(English text)-->
      |
      v
[4] TTS (Kokoro)  --(English audio chunk)-->  output queue
      |
      v
Speaker / audio output (played in order as chunks complete)
```

Each stage after VAD reuses the exact same methods already built and tested in `BaltiTarjumanPipeline` (`.transcribe()`, `.translate()`, `.synthesize()`) — the streaming layer wraps that pipeline, it doesn't reimplement it. This is the same class-reuse principle as the batch build: no forking of ASR/MT/TTS logic.

---

## Component 1: Streaming VAD Segmenter

**Problem it solves:** raw mic audio is a continuous stream; the pipeline needs discrete, complete speech segments to hand to Whisper.

**Design:**
- Maintain a rolling audio buffer.
- Feed small frames (Silero VAD's expected frame size, e.g. ~30ms) into Silero VAD in streaming mode as they arrive from the mic.
- Track speech/silence state. When VAD detects a silence gap of a set threshold (e.g. ~400-600ms) after speech, close the segment and emit everything from segment-start to segment-end.
- **Overlap padding:** include ~150-250ms of audio before segment-start and after segment-end (from the rolling buffer) when emitting, so words right at the boundary aren't clipped mid-phoneme. This directly addresses the "words cut at chunk boundaries" failure mode flagged as a risk earlier.
- **Max segment cap:** force-emit a segment if speech continues past ~6-8s even without a silence gap, so one long run-on sentence can't blow the latency budget for everything after it.

Output: a queue of `(audio_array, start_ts, end_ts)` speech segments, emitted as soon as each is detected — not waiting for the whole recording to end.

---

## Component 2: Segment Processing Worker

**Problem it solves:** ASR → MT → TTS are blocking, GPU-bound calls. They shouldn't block the VAD segmenter from continuing to listen and detect the *next* segment while the current one is still being processed.

**Design:**
- VAD segmenter runs in one thread, pushing finished segments onto an input queue.
- A separate worker thread pulls segments off that queue one at a time and runs them through `pipeline.transcribe()` → `pipeline.translate()` → `pipeline.synthesize()`, pushing the resulting audio chunk onto an output queue.
- Since there's one GPU, this worker is inherently sequential — segments are processed in the order they were spoken, which is what you want for coherent output anyway.
- If someone speaks continuously and segments arrive faster than they can be processed, the input queue simply grows — worth logging queue depth during testing so you can see if a speaker outruns processing, and decide whether to warn the user or drop/merge segments in that case.

---

## Component 3: Output Playback

**Design:**
- A third thread (or the main thread, depending on final hosting choice) pulls finished audio chunks off the output queue and plays them in order.
- Because chunks are produced sequentially and pulled in order, playback stays coherent even though processing happens segment-by-segment.

---

## Latency budget (rough, to validate empirically in Step 7)

| Stage | Expected time |
|---|---|
| VAD silence-detection delay | ~0.4-0.6s (the threshold itself) |
| Whisper transcription (short segment, GPU) | ~0.5-1.5s |
| NLLB translation | ~0.2-0.5s |
| Kokoro synthesis | ~0.5-1s |
| **Total** | **~1.6-3.6s** — within the 2-4s target |

This is an estimate, not a measurement — Step 7 (end-to-end real-time test) is where this gets validated against real hardware and real speech.

---

## Hosting / UI layer — kept separate from the core loop

Per the earlier Gradio postmortem: the core streaming loop above is framework-agnostic — plain Python generator + threads/queues, testable from a script with a synthetic `.wav` file, no UI involved. The hosting layer (Gradio, or an alternative) is a thin wrapper around it, built and tested *after* the core loop is confirmed working standalone. This way, if hosting has issues again, it's isolated and obviously not a pipeline logic bug.

**Recommendation:** stick with Gradio (you already have the components and experience with it), but apply the fixes identified from the postmortem:
- Use `.stream()` with explicit `time_limit` and `stream_every` params (not `.click()`/`.change()`).
- Make the handler a generator (`yield`, not `return`) so output streams back continuously.
- Match `gr.Audio(streaming=True, autoplay=True)` on the output side.
- Test with a pre-recorded file fed through the same code path before testing live mic input, so a mic-permission/browser issue doesn't get mistaken for a pipeline bug.

---

## What's deferred to implementation (Steps 5-6)

This doc defines shape and responsibilities, not final code. Implementation will need to decide:
- Exact Silero VAD streaming API calls (frame size, threading model specifics)
- Queue implementation (`queue.Queue` is likely sufficient — no need for anything heavier at this scale)
- Whether wav2vec2 fallback is even reachable in streaming mode, or whether Whisper failure in a live session should just skip that segment with a warning (batch mode's fallback logic may not translate cleanly to a live low-latency context — worth a decision when we get there)
