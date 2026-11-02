# Ebook to Audiobook with VoxCPM 2

Turn your eBooks (`.epub`, `.pdf`, `.mobi`, `.txt`, `.html`) into 48kHz studio-quality audiobooks using [OpenBMB's VoxCPM 2](https://github.com/OpenBMB/VoxCPM) text-to-speech and voice cloning model.

Built for [Pinokio](https://pinokio.computer/) — the easiest way to install and run AI applications locally.

## Features
* **48kHz Studio-Quality Output:** Powered by VoxCPM 2's AudioVAE V2 for native 48kHz synthesis.
* **30 Languages Supported:** Synthesize audiobooks directly across 30 languages with context-aware prosody.
* **Three Voice Modes:**
  * **Controllable Voice Cloning (Default):** Clone any voice from a short audio sample (no transcript needed) and optionally guide pacing/emotion via a natural-language Control Instruction.
  * **Ultimate Hi-Fi Cloning:** Provide both the reference audio and its transcript for continuation-grade vocal cloning.
  * **Voice Design:** Clear the reference audio and describe a brand-new voice in plain text (automatically anchored so the designed voice stays consistent across the entire book).
* **Smart Chunking & Text Normalization:** Prevents long-form drift and OOM errors while normalizing numbers, times, currencies, and abbreviations.
* **Automatic MP3 Metadata & Cover Art:** Embeds the book's title, author, and extracted cover art into the final MP3.

## How to Use
1. Click **Install** in Pinokio to set up the Python environment and pre-download `openbmb/VoxCPM2` into `hf_cache`.
2. Click **Start** to launch the Gradio Web UI.
3. Keep the default reference voice, upload your own 5–30 second voice clip, or clear the audio and enter a **Voice Style / Control Instruction**.
4. Upload one or more eBooks and click **Start Processing**. Completed audiobooks are saved in `Working_files/Book`.

## Programmatic API Usage (Gradio Client)

### Python
```python
from gradio_client import Client, handle_file

client = Client("http://127.0.0.1:7860")
result = client.predict(
    ref_audio_input=handle_file("app/default_voice.mp3"),
    control_instruction_input="Calm, warm audiobook narrator",
    use_ultimate_cloning=False,
    ref_text_input="",
    gen_file_input=[handle_file("my_book.epub")],
    speed=1.0,
    max_phrase_length=250,
    max_chunk_length=500,
    num_steps=10,
    cfg=2.0,
    seed_input=42,
    wetext_normalize=False,
    api_name="/basic_tts"
)
print("Generated audiobooks:", result)
