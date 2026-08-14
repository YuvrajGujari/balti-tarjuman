"""
test_asr_fallback.py

Pulls real held-out test clips from YuvrajGujari/balti-tarjuman-data and
uses them to:
  1. Measure Whisper's actual latency distribution (so whisper_timeout_sec
     can be set from real data instead of a guessed constant).
  2. Deliberately exercise the timeout fallback path (temporarily forcing
     a very low timeout) and the exception fallback path, to confirm both
     actually work end-to-end rather than just reading correctly.

Run:
    python test_asr_fallback.py
"""

import io
import os
import time
import statistics

import numpy as np
import soundfile as sf
import librosa
from datasets import load_dataset, Audio
from huggingface_hub import login

# NOTE: BaltiTarjumanPipeline is NOT imported here — it's assumed to
# already be defined in an earlier cell in this notebook (the one that
# ran successfully). Run that cell first, then run this one.


def authenticate(hf_token=None):
    """
    Logs in to Hugging Face Hub explicitly, independent of pipeline
    construction — needed since load_test_samples() may run before
    BaltiTarjumanPipeline() is constructed (which is where login()
    normally happens inside this project's pipeline.py).
    """
    if hf_token is None:
        # On Kaggle: Add-ons -> Secrets -> add HF_TOKEN, then it's
        # available as an env var if "Attach" is enabled for this
        # notebook. Falls back to a plain env var otherwise.
        try:
            from kaggle_secrets import UserSecretsClient
            hf_token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            hf_token = os.environ.get("HF_TOKEN")

    if not hf_token:
        raise RuntimeError(
            "No HF_TOKEN found. Set it via Kaggle Secrets (Add-ons -> "
            "Secrets -> HF_TOKEN) or os.environ['HF_TOKEN'] = '...' "
            "before running this script — balti-tarjuman-data is a "
            "private dataset and needs authentication to load."
        )
    login(token=hf_token)
    return hf_token


def load_test_samples(n=20, hf_token=None):
    """
    Loads n samples from the test split. Uses streaming so it doesn't
    download the whole dataset just to grab a handful of clips.
    """
    ds = load_dataset(
        "YuvrajGujari/balti-tarjuman-data",
        split="test",
        streaming=True,
        token=hf_token,  # explicit token, not relying on pipeline's login() timing
    )
    # Disable auto-decoding: newer `datasets` versions decode the audio
    # column into a torchcodec.AudioDecoder object automatically, which
    # isn't subscriptable like example["audio"]["bytes"] expects. This
    # is the exact torchcodec issue this project already hit and solved
    # once during ASR fine-tuning (see the Engineering Case Study) —
    # disabling decode here gets the raw bytes/path dict back instead.
    ds = ds.cast_column("audio", Audio(decode=False))

    samples = []
    for i, example in enumerate(ds):
        if i >= n:
            break
        # Decode audio bytes -> numpy array via soundfile, matching the
        # "bytes field, not path field" lesson from the ASR fine-tuning
        # phase — the same caveat applies when reading from a streamed
        # HF dataset here.
        audio_bytes = example["audio"]["bytes"]
        audio_array, sr = sf.read(io.BytesIO(audio_bytes))

        # Both the Whisper and Wav2Vec2 processors were trained on 16kHz
        # audio and will raise a ValueError otherwise. The raw dataset
        # bytes are at their original/native sample rate (e.g. 32kHz),
        # since decode=False above skips datasets' normal auto-resample
        # step — so we resample manually here instead.
        if sr != 16000:
            audio_array = librosa.resample(
                audio_array.astype(np.float32), orig_sr=sr, target_sr=16000
            )
            sr = 16000

        samples.append(
            {
                "audio_array": audio_array,
                "sr": sr,
                "reference_text": example.get("sentence", ""),
            }
        )
    return samples


def measure_whisper_latency(pipeline, samples):
    """
    Calls the raw Whisper stage directly (bypassing the fallback logic)
    to get a clean latency distribution, unaffected by any timeout.
    """
    latencies = []
    for i, sample in enumerate(samples):
        t0 = time.perf_counter()
        try:
            text = pipeline._run_whisper(sample["audio_array"], sample["sr"])
            elapsed = time.perf_counter() - t0
            latencies.append(elapsed)
            print(f"[{i}] {elapsed:.2f}s  →  {text[:60]}")
        except Exception as e:
            print(f"[{i}] Whisper raised an exception: {e}")

    if not latencies:
        print("No successful Whisper runs — can't compute latency stats.")
        return None

    latencies.sort()
    p50 = statistics.median(latencies)
    p95 = latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)]
    print("\n--- Whisper latency summary ---")
    print(f"n={len(latencies)}  min={min(latencies):.2f}s  "
          f"median={p50:.2f}s  p95={p95:.2f}s  max={max(latencies):.2f}s")
    print(f"Suggested whisper_timeout_sec (p95 + 20% margin): {p95 * 1.2:.2f}s")
    return {"p50": p50, "p95": p95, "min": min(latencies), "max": max(latencies)}


def test_timeout_fallback(samples):
    """
    Deliberately forces a near-zero timeout so every call takes the
    timeout path — confirms wav2vec2_fallback_timeout actually fires
    and returns a usable transcript, not just that the code runs.
    """
    print("\n--- Testing timeout fallback path (forced) ---")
    pipeline = BaltiTarjumanPipeline(whisper_timeout_sec=0.01)
    sample = samples[0]
    text, source, is_reliable = pipeline.transcribe(sample["audio_array"], sample["sr"])
    print(f"asr_used={source}  is_reliable={is_reliable}")
    print(f"transcript: {text}")
    assert source == "wav2vec2_fallback_timeout", (
        f"Expected wav2vec2_fallback_timeout, got {source} — "
        f"timeout path did not trigger as expected."
    )
    print("✅ Timeout fallback path confirmed working.")
    pipeline.close()


def test_normal_path(pipeline, samples, threshold_sec):
    """
    Runs with a realistic (measured) timeout to confirm the normal,
    non-fallback path still works correctly for typical clips.
    """
    print(f"\n--- Testing normal path (timeout={threshold_sec:.2f}s) ---")
    pipeline.whisper_timeout_sec = threshold_sec
    for i, sample in enumerate(samples[:5]):
        text, source, is_reliable = pipeline.transcribe(sample["audio_array"], sample["sr"])
        print(f"[{i}] asr_used={source}  transcript: {text[:60]}")


if __name__ == "__main__":
    # Sanity check: fail fast with a clear message if the earlier cell
    # defining BaltiTarjumanPipeline hasn't been run yet in this session.
    assert "BaltiTarjumanPipeline" in dir(), (
        "BaltiTarjumanPipeline is not defined. Run the earlier cell that "
        "defines the class first, then re-run this cell."
    )

    hf_token = authenticate()  # do this FIRST, before anything needs auth

    print("Loading test samples from YuvrajGujari/balti-tarjuman-data (test split)...")
    samples = load_test_samples(n=20, hf_token=hf_token)
    print(f"Loaded {len(samples)} samples.\n")

    pipeline = BaltiTarjumanPipeline()

    stats = measure_whisper_latency(pipeline, samples)

    if stats:
        suggested_timeout = stats["p95"] * 1.2
        test_normal_path(pipeline, samples, threshold_sec=suggested_timeout)

    pipeline.close()

    # Separate pipeline instance for the forced-timeout test, since we
    # want to isolate it from the latency measurement above.
    test_timeout_fallback(samples)
