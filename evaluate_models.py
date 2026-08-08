import torch
import soundfile as sf
import io
from datasets import load_dataset, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using compute device: {device}")

# 1. Load Validation Dataset
print("\n[1/3] Loading Balti Validation Dataset...")
ds = load_dataset("YuvrajGujari/balti-tarjuman-data", split="validation")
ds = ds.cast_column("audio", Audio(decode=False))

# 2. Load Both Fine-Tuned Models
print("\n[2/3] Loading Whisper-small Champion (17.40% WER)...")
whisper_id = "YuvrajGujari/whisper-small-balti"
whisper_processor = WhisperProcessor.from_pretrained(whisper_id)
whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_id).to(device)
whisper_model.eval()

print("[3/3] Loading Wav2Vec2 XLS-R Baseline (22.82% WER)...")
w2v_id = "YuvrajGujari/wav2vec2-balti"
w2v_processor = Wav2Vec2Processor.from_pretrained(w2v_id)
w2v_model = Wav2Vec2ForCTC.from_pretrained(w2v_id).to(device)
w2v_model.eval()

# 3. Inference Loop
num_samples = 10
print("\n" + "=" * 90)
print(f"{'SIMULTANEOUS MODEL INFERENCE COMPARISON':^90}")
print("=" * 90)

for idx in range(num_samples):
    example = ds[idx]
    audio_bytes = example["audio"]["bytes"]
    audio_array, sr = sf.read(io.BytesIO(audio_bytes))
    
    # Resample to 16kHz
    if sr != 16000:
        import librosa
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
    
    # --- Whisper Generation ---
    whisper_inputs = whisper_processor(
        audio_array, 
        sampling_rate=16000, 
        return_tensors="pt"
    ).input_features.to(device)
    
    with torch.no_grad():
        whisper_tokens = whisper_model.generate(
            whisper_inputs, 
            forced_decoder_ids=None
        )
    whisper_pred = whisper_processor.batch_decode(whisper_tokens, skip_special_tokens=True)[0]
    
    # --- Wav2Vec2 Generation ---
    w2v_inputs = w2v_processor(
        audio_array, 
        sampling_rate=16000, 
        return_tensors="pt"
    ).input_values.to(device)
    
    with torch.no_grad():
        logits = w2v_model(w2v_inputs).logits
    w2v_tokens = torch.argmax(logits, dim=-1)
    w2v_pred = w2v_processor.batch_decode(w2v_tokens)[0]
    
    # Print Results
    print(f"\nSample #{idx + 1}")
    print(f"📌 Ground Truth : {example['text']}")
    print(f"🏆 Whisper 17.4% : {whisper_pred}")
    print(f"⚡ Wav2Vec2 22.8%: {w2v_pred}")
    print("-" * 90)
