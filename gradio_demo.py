"""
Gradio demo for Balti Tarjuman — supports both live microphone input
and uploaded audio files, using the same BaltiTarjumanPipeline either way.

Run:
    python gradio_demo.py
"""

import torch
import numpy as np
import soundfile as sf
import gradio as gr

from pipeline import BaltiTarjumanPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = BaltiTarjumanPipeline(device=device)


def translate_audio(audio):
    """
    Handles input from BOTH the microphone and the upload source,
    since Gradio hands both back in the same format.

    `audio` is a tuple: (sample_rate, numpy_array) when using type="numpy",
    or a filepath string when using type="filepath". This uses type="numpy"
    for consistency whether the audio came from mic or upload.
    """
    if audio is None:
        return "No audio provided.", None

    sample_rate, audio_array = audio

    # Gradio's numpy audio can come in as int16 — normalize to float32 in [-1, 1]
    # since that's what most ASR/VAD pipelines expect.
    if audio_array.dtype != np.float32:
        audio_array = audio_array.astype(np.float32) / np.iinfo(np.int16).max

    # If stereo, collapse to mono
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)

    # Write to a temp wav file since BaltiTarjumanPipeline.translate()
    # expects a file path (matches the batch pipeline's existing interface —
    # no need to change pipeline.py itself).
    temp_path = "temp_input.wav"
    sf.write(temp_path, audio_array, samplerate=sample_rate)

    translated_text, synthesized_audio = pipeline.translate(temp_path)

    # Gradio's audio output expects (sample_rate, numpy_array)
    output_sample_rate = 24000  # matches Kokoro's output rate
    return translated_text, (output_sample_rate, synthesized_audio)


with gr.Blocks(title="Balti Tarjuman") as demo:
    gr.Markdown("# 🏔️ Balti Tarjuman — Balti → English Speech Translation")
    gr.Markdown(
        "Speak into your microphone **or** upload a Balti audio file. "
        "The pipeline runs VAD → ASR (Whisper) → MT (NLLB) → TTS (Kokoro)."
    )

    with gr.Row():
        audio_input = gr.Audio(
            sources=["microphone", "upload"],
            type="numpy",
            label="Balti Audio (record or upload)",
        )

    submit_btn = gr.Button("Translate", variant="primary")

    with gr.Row():
        text_output = gr.Textbox(label="English Translation")
        audio_output = gr.Audio(label="English Speech Output", autoplay=True)

    submit_btn.click(
        fn=translate_audio,
        inputs=audio_input,
        outputs=[text_output, audio_output],
    )

    # Also trigger on file change (covers upload without needing the button,
    # while still requiring the button for a fresh mic recording)
    audio_input.change(
        fn=translate_audio,
        inputs=audio_input,
        outputs=[text_output, audio_output],
    )

if __name__ == "__main__":
    demo.launch()
