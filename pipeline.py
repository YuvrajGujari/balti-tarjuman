
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_TOKEN")

from huggingface_hub import login
login(token=hf_token)

# %%
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
# %% VAD — Silero
vad_model, vad_utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False)
(get_speech_timestamps, _, read_audio, _, _) = vad_utils
# %% Primary ASR — fine-tuned Whisper
from transformers import WhisperForConditionalGeneration, WhisperProcessor

whisper_model = WhisperForConditionalGeneration.from_pretrained("YuvrajGujari/whisper-small-balti").to(device)
whisper_processor = WhisperProcessor.from_pretrained("YuvrajGujari/whisper-small-balti")
# %% Backup ASR — MMS (pretrained, not fine-tuned; bft not covered, falls back to bod — known unreliable, see below)
from transformers import Wav2Vec2ForCTC, AutoProcessor

mms_model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all").to(device)
mms_processor = AutoProcessor.from_pretrained("facebook/mms-1b-all")

MMS_TARGET_LANG = "bft"
MMS_IS_RELIABLE = True
try:
    mms_processor.tokenizer.set_target_lang(MMS_TARGET_LANG)
    mms_model.load_adapter(MMS_TARGET_LANG)
    print(f"MMS: '{MMS_TARGET_LANG}' adapter loaded successfully.")
except Exception as e:
    MMS_TARGET_LANG = "bod"
    MMS_IS_RELIABLE = False  # confirmed via manual eval: wrong script (Tibetan) vs. our Nastaliq data, low quality
    print(f"MMS: 'bft' adapter unavailable. Falling back to '{MMS_TARGET_LANG}' (KNOWN UNRELIABLE for this language).")
    mms_processor.tokenizer.set_target_lang(MMS_TARGET_LANG)
    mms_model.load_adapter(MMS_TARGET_LANG)
# %% Translation — fine-tuned NLLB
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

mt_tokenizer = AutoTokenizer.from_pretrained("YuvrajGujari/nllb-balti-mt")
mt_model = AutoModelForSeq2SeqLM.from_pretrained("YuvrajGujari/nllb-balti-mt").to(device)
mt_tokenizer.src_lang = "bft_Arab"
# %% TTS — Kokoro-82M
import subprocess
subprocess.run(["pip", "install", "-q", "kokoro"], check=False)
from kokoro import KPipeline

tts_pipeline = KPipeline(lang_code='a')
# %% Core pipeline function
import soundfile as sf

def transcribe_with_failover(audio_array, sr=16000):
    """Try Whisper first. On failure, attempt MMS but flag output as unreliable
    rather than silently passing degraded/wrong-script text downstream."""
    try:
        input_features = whisper_processor.feature_extractor(
            audio_array, sampling_rate=sr
        ).input_features
        input_features = torch.tensor(input_features).to(device)
        predicted_ids = whisper_model.generate(input_features)
        text = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return text, "whisper", True
    except Exception as e:
        print(f"Whisper failed ({e}) — attempting MMS fallback (lang={MMS_TARGET_LANG}).")
        inputs = mms_processor(audio_array, sampling_rate=sr, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = mms_model(**inputs).logits
        ids = torch.argmax(logits, dim=-1)[0]
        text = mms_processor.decode(ids)
        return text, "mms_fallback", MMS_IS_RELIABLE

def translate(balti_text):
    inputs = mt_tokenizer(balti_text, return_tensors="pt").to(device)
    generated = mt_model.generate(
        **inputs,
        forced_bos_token_id=mt_tokenizer.convert_tokens_to_ids("eng_Latn"),
        max_length=128,
    )
    return mt_tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

def synthesize_speech(english_text, output_path="output.wav"):
    generator = tts_pipeline(english_text, voice='af_heart')
    for _, _, audio in generator:
        sf.write(output_path, audio, 24000)
        return output_path
def run_pipeline(input_audio_path, output_audio_path="pipeline_output.wav"):
    wav = read_audio(input_audio_path, sampling_rate=16000)
    speech_timestamps = get_speech_timestamps(wav, vad_model, sampling_rate=16000)
    print(f"VAD found {len(speech_timestamps)} speech segment(s).")

    full_audio = wav.numpy()
    balti_text, asr_source, is_reliable = transcribe_with_failover(full_audio, sr=16000)
    print(f"[{asr_source}] Balti transcript: {balti_text}")

    if not is_reliable:
        print("WARNING: ASR output flagged unreliable (fallback used, known script/language mismatch). "
              "Skipping translation to avoid compounding errors.")
        return {
            "balti_text": balti_text,
            "english_text": None,
            "audio_path": None,
            "asr_used": asr_source,
            "warning": "fallback_unreliable",
        }

    english_text = translate(balti_text)
    print(f"English translation: {english_text}")

    out_path = synthesize_speech(english_text, output_audio_path)
    print(f"Output audio written to: {out_path}")
    return {
        "balti_text": balti_text,
        "english_text": english_text,
        "audio_path": out_path,
        "asr_used": asr_source,
    }

# %% Test run — replace with a real Balti test clip
# result = run_pipeline("path/to/test_clip.wav")
# print(result)

