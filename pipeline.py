"""
BaltiTarjumanPipeline — Balti speech-to-English speech translation.

Silero VAD -> Whisper-small (primary ASR) + wav2vec2-XLS-R (backup ASR)
-> NLLB-200-distilled-600M (MT) -> Kokoro-82M (TTS)

Usage:
    pipeline = BaltiTarjumanPipeline()
    result = pipeline.run("path/to/audio.wav")
    print(result["english_text"])
"""

import os
import subprocess

import torch
import soundfile as sf
from huggingface_hub import login
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Wav2Vec2ForCTC,
    AutoProcessor,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

WHISPER_MODEL_ID = "YuvrajGujari/whisper-small-balti"
WAV2VEC2_MODEL_ID = "YuvrajGujari/wav2vec2-balti-specaugment"  # 21.11% WER, cold-start + SpecAugment retrain
NLLB_MODEL_ID = "YuvrajGujari/nllb-balti-mt"


class BaltiTarjumanPipeline:
    """
    Loads all models once at construction time, then exposes stage-level
    methods (transcribe / translate / synthesize) plus a single run()
    entry point for the full batch pipeline.
    """

    def __init__(self, hf_token=None, device=None):
        if hf_token is None:
            hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            login(token=hf_token)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"BaltiTarjumanPipeline: using device={self.device}")

        self._load_vad()
        self._load_asr()
        self._load_mt()
        self._load_tts()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_vad(self):
        self.vad_model, vad_utils = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", force_reload=False
        )
        (self._get_speech_timestamps, _, self._read_audio, _, _) = vad_utils

    def _load_asr(self):
        # Primary ASR — fine-tuned Whisper
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
            WHISPER_MODEL_ID
        ).to(self.device)
        self.whisper_processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_ID)

        # Backup ASR — project's own fine-tuned wav2vec2 (not generic MMS)
        self.wav2vec2_model = Wav2Vec2ForCTC.from_pretrained(
            WAV2VEC2_MODEL_ID
        ).to(self.device)
        self.wav2vec2_processor = AutoProcessor.from_pretrained(WAV2VEC2_MODEL_ID)

    def _load_mt(self):
        self.mt_tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_ID)
        self.mt_model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_ID).to(
            self.device
        )
        self.mt_tokenizer.src_lang = "bft_Arab"

    def _load_tts(self):
        subprocess.run(["pip", "install", "-q", "kokoro"], check=False)
        from kokoro import KPipeline

        self.tts_pipeline = KPipeline(lang_code="a")

    # ------------------------------------------------------------------
    # Stage methods
    # ------------------------------------------------------------------

    def detect_speech(self, input_audio_path, sr=16000):
        """Run VAD, return (waveform, speech_timestamps)."""
        wav = self._read_audio(input_audio_path, sampling_rate=sr)
        timestamps = self._get_speech_timestamps(wav, self.vad_model, sampling_rate=sr)
        return wav, timestamps

    def transcribe(self, audio_array, sr=16000):
        """
        Try Whisper first (primary). On failure, fall back to the project's
        fine-tuned wav2vec2 backup. Returns (text, source, is_reliable).
        Both models were fine-tuned specifically on this project's Balti
        data, so both are treated as reliable — unlike the earlier
        MMS-based fallback, which was flagged unreliable due to a
        script/language mismatch.
        """
        try:
            input_features = self.whisper_processor.feature_extractor(
                audio_array, sampling_rate=sr
            ).input_features
            input_features = torch.tensor(input_features).to(self.device)
            predicted_ids = self.whisper_model.generate(input_features)
            text = self.whisper_processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]
            return text, "whisper", True
        except Exception as e:
            print(f"Whisper failed ({e}) — falling back to wav2vec2 backup ASR.")
            inputs = self.wav2vec2_processor(
                audio_array, sampling_rate=sr, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                logits = self.wav2vec2_model(**inputs).logits
            ids = torch.argmax(logits, dim=-1)[0]
            text = self.wav2vec2_processor.decode(ids)
            return text, "wav2vec2_fallback", True

    def translate(self, balti_text):
        inputs = self.mt_tokenizer(balti_text, return_tensors="pt").to(self.device)
        generated = self.mt_model.generate(
            **inputs,
            forced_bos_token_id=self.mt_tokenizer.convert_tokens_to_ids("eng_Latn"),
            max_length=128,
        )
        return self.mt_tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

    def synthesize(self, english_text, output_path="output.wav"):
        generator = self.tts_pipeline(english_text, voice="af_heart")
        for _, _, audio in generator:
            sf.write(output_path, audio, 24000)
            return output_path
        return None

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(self, input_audio_path, output_audio_path="pipeline_output.wav"):
        wav, speech_timestamps = self.detect_speech(input_audio_path)
        print(f"VAD found {len(speech_timestamps)} speech segment(s).")

        full_audio = wav.numpy()
        balti_text, asr_source, is_reliable = self.transcribe(full_audio, sr=16000)
        print(f"[{asr_source}] Balti transcript: {balti_text}")

        english_text = self.translate(balti_text)
        print(f"English translation: {english_text}")

        out_path = self.synthesize(english_text, output_audio_path)
        print(f"Output audio written to: {out_path}")

        return {
            "balti_text": balti_text,
            "english_text": english_text,
            "audio_path": out_path,
            "asr_used": asr_source,
        }


if __name__ == "__main__":
    # Quick smoke test — replace with a real Balti test clip
    pipeline = BaltiTarjumanPipeline()
    result = pipeline.run("path/to/test_clip.wav")
    print(result)
