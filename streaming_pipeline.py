"""
StreamingBaltiTarjumanPipeline — real-time streaming layer over BaltiTarjumanPipeline.

Wraps VAD -> ASR -> MT -> TTS with threaded segmentation and processing.
Reuses BaltiTarjumanPipeline's already-loaded models (including its VAD
model/VADIterator); does not duplicate model-loading logic.

Usage:
    from pipeline import BaltiTarjumanPipeline
    from streaming_pipeline import StreamingBaltiTarjumanPipeline

    pipeline = BaltiTarjumanPipeline()
    streaming = StreamingBaltiTarjumanPipeline(pipeline)
    streaming.start()
    streaming.feed_wav_file("path/to/audio.wav")
"""

import queue
import threading
import time

import numpy as np
import soundfile as sf
import torch

from pipeline import BaltiTarjumanPipeline


class VADSegmenter:
    """Turns a stream of raw audio frames into complete speech segments."""

    def __init__(
        self,
        vad_model,
        vad_iterator_cls,
        sample_rate=16000,
        frame_size=512,
        threshold=0.3,
        min_silence_duration_ms=500,
        speech_pad_ms=200,
        max_segment_seconds=8.0,
    ):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.vad_iterator = vad_iterator_cls(
            vad_model,
            threshold=threshold,
            sampling_rate=sample_rate,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        )
        self.max_segment_samples = int(max_segment_seconds * sample_rate)
        self._buffer = []
        self._in_speech = False

    def process_frame(self, frame: np.ndarray):
        tensor = torch.from_numpy(frame)
        result = self.vad_iterator(tensor, return_seconds=False)

        if self._in_speech:
            self._buffer.append(frame)

        emitted = None

        if result is not None:
            if "start" in result:
                self._in_speech = True
                self._buffer = [frame]
            if "end" in result:
                self._in_speech = False
                if self._buffer:
                    emitted = np.concatenate(self._buffer)
                self._buffer = []

        # Force-close on max duration so one long run-on sentence can't
        # blow the latency budget for everything spoken after it.
        if self._in_speech and self._buffer:
            current_len = sum(len(b) for b in self._buffer)
            if current_len >= self.max_segment_samples:
                emitted = np.concatenate(self._buffer)
                self._buffer = []
                self._in_speech = False
                self.vad_iterator.reset_states()

        return emitted

    def flush(self):
        """Force-emit any pending buffered segment — call at end of stream.
        Handles the case where trailing silence is just barely over
        min_silence_duration_ms and the "end" event never fires before
        the stream runs out."""
        if self._in_speech and self._buffer:
            emitted = np.concatenate(self._buffer)
            self._buffer = []
            self._in_speech = False
            self.vad_iterator.reset_states()
            return emitted
        return None

    def reset(self):
        self.vad_iterator.reset_states()
        self._buffer = []
        self._in_speech = False


class StreamingBaltiTarjumanPipeline:
    """
    Wraps a BaltiTarjumanPipeline instance with streaming segmentation and
    threaded processing. Reuses .transcribe() / .translate() /
    .synthesize_array() and the already-loaded VAD model as-is — no model
    logic duplicated here.
    """

    def __init__(self, pipeline=None, frame_size=512, sample_rate=16000):
        self.pipeline = pipeline or BaltiTarjumanPipeline()
        self.sample_rate = sample_rate
        self.frame_size = frame_size

        self.segmenter = VADSegmenter(
            vad_model=self.pipeline.vad_model,
            vad_iterator_cls=self.pipeline.VADIterator,
            sample_rate=sample_rate,
            frame_size=frame_size,
        )

        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread = None

    def start(self):
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def stop(self):
        self._stop_event.set()
        self.input_queue.put(None)  # sentinel to unblock a queue.get()
        if self._worker_thread:
            self._worker_thread.join(timeout=5)

    def feed_audio_frame(self, frame: np.ndarray):
        """Call with each raw audio frame (length == frame_size, sample_rate Hz)."""
        segment = self.segmenter.process_frame(frame)
        if segment is not None:
            self.input_queue.put(segment)

    def flush(self):
        """Force-process whatever's still buffered — call at end of a file/session."""
        segment = self.segmenter.flush()
        if segment is not None:
            self.input_queue.put(segment)

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                segment = self.input_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if segment is None:
                break

            t0 = time.time()
            balti_text, asr_source, _ = self.pipeline.transcribe(
                segment, sr=self.sample_rate
            )
            english_text = self.pipeline.translate(balti_text)
            audio_array, audio_sr = self.pipeline.synthesize_array(english_text)
            latency = time.time() - t0

            self.output_queue.put(
                {
                    "balti_text": balti_text,
                    "english_text": english_text,
                    "audio_array": audio_array,
                    "audio_sr": audio_sr,
                    "asr_used": asr_source,
                    "processing_latency_sec": latency,
                }
            )

    def get_next_output(self, timeout=None):
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def feed_wav_file(self, path, realtime=False, reset_first=True):
        """
        Standalone test helper: reads a wav file and feeds it frame-by-frame
        as if it were live mic input.

        reset_first=True (default) clears segmenter state before reading
        this file — important when feeding multiple files back-to-back,
        since VADIterator carries state across calls.
        """
        if reset_first:
            self.segmenter.reset()

        audio, sr = sf.read(path, dtype="float32")
        if sr != self.sample_rate:
            raise ValueError(
                f"File is {sr}Hz, pipeline expects {self.sample_rate}Hz. Resample first."
            )
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        frame_duration = self.frame_size / self.sample_rate

        for i in range(0, len(audio), self.frame_size):
            frame = audio[i : i + self.frame_size]
            if len(frame) < self.frame_size:
                frame = np.pad(frame, (0, self.frame_size - len(frame)))
            self.feed_audio_frame(frame)
            if realtime:
                time.sleep(frame_duration)

        self.flush()  # ensure a segment right at end-of-file isn't lost
