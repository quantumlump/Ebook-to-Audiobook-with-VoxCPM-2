import os
import shutil
import tempfile
import time
import io
import re
import gc
import subprocess
import warnings
import starlette.datastructures
import traceback
import uuid
import inspect

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- 1. THE PINOKIO WATCHER CRASH FIX ---
SYSTEM_TEMP = tempfile.gettempdir()
LOCAL_TEMP = os.path.join(SYSTEM_TEMP, "voxcpm_gradio_temp")

if os.path.exists(LOCAL_TEMP):
    try:
        shutil.rmtree(LOCAL_TEMP, ignore_errors=True)
    except Exception:
        pass
os.makedirs(LOCAL_TEMP, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = LOCAL_TEMP

# --- 2. THE FASTAPI RAM SPOOL FIX ---
try:
    starlette.datastructures.UploadFile.spool_max_size = 100 * 1024 * 1024
except Exception:
    pass

from num2words import num2words
from decimal import Decimal

import gradio as gr
import numpy as np
import soundfile as sf
import torch
import torchaudio

# Suppress torch._dynamo / torch.compile errors on Windows & Gradio worker threads
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass

from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize

# --- Download NLTK tokenizers if missing ---
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading required NLTK tokenizers...")
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except Exception as e:
        print(f"NLTK download warning: {e}. If offline, ensure punkt is pre-installed.")

# --- Zero Dependency Extractors & Tools ---
import fitz  # PyMuPDF
import mobi
from ebooklib import epub, ITEM_DOCUMENT, ITEM_COVER, ITEM_IMAGE
import imageio_ffmpeg
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

# --- 3. TORCHAUDIO / TORCHCODEC FALLBACK SAFEGUARD ---
_orig_torchaudio_load = torchaudio.load

def _safe_torchaudio_load(filepath, *args, **kwargs):
    try:
        return _orig_torchaudio_load(filepath, *args, **kwargs)
    except Exception:
        wav_path = str(filepath)
        try:
            data, sr = sf.read(wav_path, dtype="float32", always_2d=True)
            tensor = torch.from_numpy(data.T.copy())
            return tensor, sr
        except Exception:
            tmp_wav = os.path.join(LOCAL_TEMP, f"ta_fallback_{uuid.uuid4().hex[:8]}.wav")
            try:
                subprocess.run(
                    [FFMPEG_EXE, "-y", "-i", wav_path, "-ac", "1", "-ar", "48000", tmp_wav],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                data, sr = sf.read(tmp_wav, dtype="float32", always_2d=True)
                tensor = torch.from_numpy(data.T.copy())
                return tensor, sr
            finally:
                if os.path.exists(tmp_wav):
                    try:
                        os.remove(tmp_wav)
                    except OSError:
                        pass

torchaudio.load = _safe_torchaudio_load

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)

# --- VoxCPM 2 Engine Imports ---
from voxcpm import VoxCPM

if torch.cuda.is_available():
    device = "cuda"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

device = os.environ.get("VOXCPM_DEVICE", device)
print(f"Using Device: {device}")

# --- Load VoxCPM 2 Model ---
CHECKPOINT = os.environ.get("VOXCPM_MODEL", "openbmb/VoxCPM2")
OPTIMIZE_COMPILE = os.environ.get("VOXCPM_OPTIMIZE", "0").lower() in ("1", "true", "yes")

print(f"Loading VoxCPM 2 model from {CHECKPOINT} to {device} (optimize={OPTIMIZE_COMPILE}) ...")
try:
    model = VoxCPM.from_pretrained(
        hf_model_id=CHECKPOINT,
        load_denoiser=False,
        optimize=OPTIMIZE_COMPILE,
        device=device,
        local_files_only=True,
    )
except Exception:
    print("Local cache not complete yet; downloading from Hugging Face Hub...")
    try:
        model = VoxCPM.from_pretrained(
            hf_model_id=CHECKPOINT,
            load_denoiser=False,
            optimize=OPTIMIZE_COMPILE,
            device=device,
            local_files_only=False,
        )
    except Exception as e:
        if device == "mps":
            print(f"MPS initialization failed ({e}), falling back to CPU...")
            device = "cpu"
            model = VoxCPM.from_pretrained(
                hf_model_id=CHECKPOINT,
                load_denoiser=False,
                optimize=False,
                device="cpu",
                local_files_only=False,
            )
        else:
            raise

sampling_rate = int(getattr(model.tts_model, "sample_rate", 48000))
print(f"VoxCPM 2 Model loaded successfully! Native Sample Rate: {sampling_rate} Hz")


class EbookProgressUpdater:
    def __init__(self, progress, num_total_chunks, ebook_idx, num_ebooks, start_time):
        self._progress = progress
        self.num_total_chunks = num_total_chunks
        self.ebook_idx = ebook_idx
        self.num_ebooks = num_ebooks
        self.start_time = start_time
        self.current_chunk_idx = 0

    def set_chunk_index(self, i):
        self.current_chunk_idx = i

    def __call__(self, value, desc=None):
        chunk_percent = 0.0
        if desc:
            match = re.search(r'(\d+\.?\d*)%', desc)
            if match:
                try:
                    chunk_percent = float(match.group(1))
                except ValueError:
                    chunk_percent = 0.0

        book_progress_fraction = (self.current_chunk_idx + (chunk_percent / 100.0)) / self.num_total_chunks
        book_progress_percent = book_progress_fraction * 100
        elapsed_seconds = time.time() - self.start_time

        elapsed_hours = int(elapsed_seconds // 3600)
        elapsed_minutes = int((elapsed_seconds % 3600) // 60)
        elapsed_secs = int(elapsed_seconds % 60)
        elapsed_str = f"{elapsed_hours:02d}:{elapsed_minutes:02d}:{elapsed_secs:02d}"

        etr_str = "Calculating..."
        if elapsed_seconds > 2 and book_progress_fraction > 0.001:
            try:
                total_estimated_time = elapsed_seconds / book_progress_fraction
                etr_seconds = max(0, total_estimated_time - elapsed_seconds)
                etr_hours = int(etr_seconds // 3600)
                etr_minutes = int((etr_seconds % 3600) // 60)
                etr_secs = int(etr_seconds % 60)
                etr_str = f"{etr_hours:02d}:{etr_minutes:02d}:{etr_secs:02d}"
            except ZeroDivisionError:
                etr_str = "Calculating..."

        if self.num_ebooks > 1:
            final_desc = f"Current Book: {self.ebook_idx + 1}/{self.num_ebooks} ({book_progress_percent:.1f}%) | Chunk {self.current_chunk_idx + 1}/{self.num_total_chunks} | Elapsed: {elapsed_str} | ETR: {etr_str}"
        else:
            final_desc = f"Chunk {self.current_chunk_idx + 1}/{self.num_total_chunks} | Elapsed: {elapsed_str} | ETR: {etr_str}"

        self._progress(value, desc=final_desc)


def convert_numbers_to_words(txt):
    txt = re.sub(r'\b(1[0-9]{3}|20[0-9]{2})\b', lambda m: num2words(int(m.group(0)), to='year'), txt)
    txt = re.sub(r'(\d+)(st|nd|rd|th)\b', lambda m: num2words(int(m.group(1)), to='ordinal'), txt, flags=re.IGNORECASE)

    def number_replacer(match):
        num_str = match.group(0).replace(',', '')
        try:
            return num2words(Decimal(num_str))
        except Exception:
            return num_str

    number_pattern = r'\b\d{1,3}(?:,\d{3})*\.\d+\b|\b\d{1,3}(?:,\d{3})*\b'
    txt = re.sub(number_pattern, number_replacer, txt)
    return txt


def strip_footnotes(text: str) -> str:
    text = re.sub(r'\[\d+(?:,\s*\d+|-\d+)*\]', '', text)
    text = re.sub(r'\[(?:sic|ibid|ref|page\s\d+)\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?<=\w)[\*\†\‡\§]\d*', '', text)
    text = re.sub(r'^\s*[\*\†\‡\§]\d*\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'(?<=[a-zA-Z])\.\d{1,3}\b', '.', text)
    text = re.sub(r'\(\s*[A-Z][a-z]+,?\s+\d{4}\s*\)', '', text)
    text = re.sub(r'^\s*\d+\b\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def disambiguate_gravity(text: str) -> str:
    text = re.sub(r'\b(\d+(?:\.\d+)?)\s*(?:gs|Gs|g\'s|G\'s)\b', r'\1 Gs', text)

    def uppercase_g_replacer(match):
        num = match.group(1)
        following_text = text[match.end():match.end()+40].lower()
        network_words = ["network", "cellular", "tower", "data", "phone", "signal", "lte", "wifi", "tech", "generation"]
        if any(word in following_text for word in network_words):
            return f"{num} G"
        return f"{num} Gs"

    text = re.sub(r'\b(\d+(?:\.\d+)?)\s*G\b', uppercase_g_replacer, text)

    gravity_verbs = ["pull", "sustained", "experience", "gravity", "accelerat", "withstand", "turn", "maneuver", "burn", "thrust", "force", "vector", "pilot"]

    def lowercase_g_replacer(match):
        num = match.group(1)
        start_pos = max(0, match.start() - 60)
        end_pos = min(len(text), match.end() + 60)
        context = text[start_pos:end_pos].lower()

        if any(verb in context for verb in gravity_verbs):
            return f"{num} Gs"
        return f"{num} grams"

    text = re.sub(r'\b(\d+(?:\.\d+)?)\s*g\b', lowercase_g_replacer, text)
    return text


def disambiguate_meters_and_million(text: str) -> str:
    million_nouns = ["copies", "views", "users", "people", "subscribers", "dollars", "pounds", "euros", "yen", "members", "citizens", "inhabitants"]

    def m_replacer(match):
        num = match.group(1)
        unit = match.group(2)
        following_text = text[match.end():match.end()+40].lower()

        if any(noun in following_text for noun in million_nouns):
            return f"{num} million"
        if unit == 'M':
            return f"{num} million"
        return f"{num} meters"

    text = re.sub(r'\b(\d+(?:\.\d+)?)\s*(m|M)\b', m_replacer, text)
    return text


def clean_and_normalize_text(raw_text: str) -> str:
    def normalize_times(t_text: str) -> str:
        time_colon_pattern = r'\b(\d{1,2}):(\d{2})\s*(a\.m\.(?!\w)|p\.m\.(?!\w)|am\b|pm\b)?'

        def time_colon_replacer(match):
            hours = int(match.group(1))
            minutes = int(match.group(2))
            suffix = match.group(3)

            if hours < 1 or hours > 24 or minutes < 0 or minutes > 59:
                return match.group(0)

            hours_word = num2words(hours)

            if minutes == 0:
                minutes_word = "o'clock"
            elif minutes < 10:
                minutes_word = f"oh {num2words(minutes)}"
            else:
                minutes_word = num2words(minutes)

            suffix_word = ""
            if suffix:
                clean_suffix = suffix.lower().replace(".", "").strip()
                if clean_suffix == "am":
                    suffix_word = " ay em"
                elif clean_suffix == "pm":
                    suffix_word = " pee em"

            return f"{hours_word} {minutes_word}{suffix_word}"

        t_text = re.sub(time_colon_pattern, time_colon_replacer, t_text, flags=re.IGNORECASE)

        time_standalone_pattern = r'\b(\d{1,2})\s*(a\.m\.(?!\w)|p\.m\.(?!\w)|am\b|pm\b)'

        def time_standalone_replacer(match):
            hours = int(match.group(1))
            suffix = match.group(2)

            if hours < 1 or hours > 24:
                return match.group(0)

            hours_word = num2words(hours)
            clean_suffix = suffix.lower().replace(".", "").strip()
            suffix_word = " ay em" if clean_suffix == "am" else " pee em"

            return f"{hours_word}{suffix_word}"

        t_text = re.sub(time_standalone_pattern, time_standalone_replacer, t_text, flags=re.IGNORECASE)
        return t_text

    text = raw_text
    text = normalize_times(text)

    text = re.sub(r'\*\s*\d+\b', '', text)
    text = re.sub(r'\[[^\]]*\]', '', text)
    academic_terms = r'\b(?:spp?\.?|ssp\.?|subsp\.?|var\.?|f\.?|cf\.?|e\.g\.?|i\.e\.?|viz\.?|see|fig\.?|plate|chapter|probably)\b'
    year_pattern = r'\d{4}'
    combined_pattern = rf'\([^)]*(?:{academic_terms}|{year_pattern})[^)]*\)'
    text = re.sub(combined_pattern, '', text, flags=re.IGNORECASE)
    text = re.sub(r'#(\d)', r'number \1', text)
    text = re.sub(r'#([a-zA-Z])', r'hashtag \1', text)
    text = text.replace('#', ' pound ')

    def currency_replacer(match):
        symbol = match.group(1)
        amount = match.group(2)
        multiplier = match.group(3)

        mapping = {'$': 'dollar', '£': 'pound', '€': 'euro', '¥': 'yen'}
        curr_name = mapping.get(symbol, 'dollars')

        if multiplier:
            multiplier = multiplier.lower().strip()
            shorthand_map = {'m': 'million', 'b': 'billion', 'k': 'thousand'}
            multiplier = shorthand_map.get(multiplier, multiplier)

        if amount != "1" or multiplier:
            curr_name += "s"

        if multiplier:
            return f"{amount} {multiplier} {curr_name}"
        else:
            return f"{amount} {curr_name}"

    currency_pattern = r'([$£€¥])(\d+(?:,\d{3})*(?:\.\d+)?)\s*(m\b|M\b|b\b|B\b|k\b|K\b|million|billion|trillion|thousand|hundred)?'
    text = re.sub(currency_pattern, currency_replacer, text)

    symbol_map = {'—': ', ', '–': ', ', '&': ' and ', '%': ' percent ', '@': ' at ', 'µm': ' micrometers ', '°': ' degrees ', '+': ' plus ', '=': ' equals ', '/': ' or '}
    for old, new in symbol_map.items():
        text = text.replace(old, new)

    text = re.sub(r'\bSt\.\s+(?=[A-Z])', 'Saint ', text)
    text = re.sub(r'(?<=\d)\s*St\.\b', ' Street', text, flags=re.IGNORECASE)
    text = re.sub(r'(?<=[A-Z][a-z])\s+St\.\b', ' Street', text)

    safe_abbreviations = {"Mr.": "Mister", "Mrs.": "Missus", "Ms.": "Miss", "Dr.": "Doctor", "Prof.": "Professor", "Rev.": "Reverend", "Hon.": "Honorable", "Jr.": "Junior", "Sr.": "Senior", "Gen.": "General", "Adm.": "Admiral", "Capt.": "Captain", "Cmdr.": "Commander", "Lt.": "Lieutenant", "Sgt.": "Sergeant", "Co.": "Company", "Corp.": "Corporation", "Inc.": "Incorporated", "Ltd.": "Limited", "LLC": "Limited Liability Company", "vs.": "versus", "et al.": "et alia", "etc.": "et cetera", "e.g.": "for example", "i.e.": "that is", "Ph.D.": "Doctor of Philosophy", "M.A.": "Master of Arts", "B.A.": "Bachelor of Arts", "pp.": "pages", "vol.": "volume", "U.S.": "United States", "U.S.A.": "United States of America", "U.K.": "United Kingdom", "E.U.": "European Union", "Ave.": "Avenue", "Blvd.": "Boulevard", "Rd.": "Road", "sq.": "square", "cu.": "cubic", "deg.": "degrees", "A.M.": "ay em", "P.M.": "pee em", "Jan.": "January", "Feb.": "February", "Mar.": "March", "Apr.": "April", "Jun.": "June", "Jul.": "July", "Aug.": "August", "Sep.": "September", "Oct.": "October", "Nov.": "November", "Dec.": "December", "approx.": "approximately", "dept.": "department", "apt.": "apartment", "est.": "established"}

    text = disambiguate_gravity(text)
    text = disambiguate_meters_and_million(text)

    unit_abbreviations = {
        "mm": "millimeters", "cm": "centimeters", "km": "kilometers",
        "mg": "milligrams", "kg": "kilograms", "ft": "feet", "yd": "yards",
        "mi": "miles", "oz": "ounces", "lb": "pounds", "lbs": "pounds",
        "mph": "miles per hour", "kph": "kilometers per hour"
    }
    for abbr, full in unit_abbreviations.items():
        text = re.sub(rf'(?<=\d)\s*{re.escape(abbr)}\b', f' {full}', text, flags=re.IGNORECASE)

    for abbr, full in safe_abbreviations.items():
        text = re.sub(r'\b' + re.escape(abbr) + r'(?!\w)', full, text, flags=re.IGNORECASE)

    text = re.sub(r'\bp\.\s*(\d+)\b', r'page \1', text, flags=re.IGNORECASE)

    cleaned_text = convert_numbers_to_words(text)
    cleaned_text = cleaned_text.replace('…', '.')
    cleaned_text = cleaned_text.replace('"', ' ').replace('“', ' ').replace('”', ' ')
    cleaned_text = cleaned_text.replace('()', '').replace('( )', '')
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text


# --- Extractors (Covers & Metadata) ---
def get_epub_meta(book, namespace, name):
    meta = book.get_metadata(namespace, name)
    if meta and len(meta) > 0 and isinstance(meta[0], tuple) and len(meta[0]) > 0:
        return meta[0][0]
    return None


def read_epub(file_path):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        book = epub.read_epub(file_path)
        text_content = []

        title = get_epub_meta(book, 'DC', 'title')
        author = get_epub_meta(book, 'DC', 'creator')

        cover_bytes = None
        try:
            for item in book.get_items_of_type(ITEM_COVER):
                cover_bytes = item.get_content()
                break

            if not cover_bytes:
                for item in book.get_items_of_type(ITEM_IMAGE):
                    name = item.get_name().lower()
                    item_id = getattr(item, 'id', '').lower()
                    if 'cover' in name or 'cover' in item_id:
                        cover_bytes = item.get_content()
                        break
        except Exception as e:
            print(f"EPUB cover extraction warning: {e}")

        for item in book.get_items_of_type(ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            for unwanted in soup(["aside", "footnote", "reftag", "nav", "footer"]):
                unwanted.decompose()
            text_content.append(soup.get_text(separator=' ', strip=True))

        return ' '.join(text_content), title, author, cover_bytes


def read_pdf(file_path):
    doc = fitz.open(file_path)
    title = doc.metadata.get("title")
    author = doc.metadata.get("author")

    cover_bytes = None
    try:
        if len(doc) > 0:
            page = doc.load_page(0)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            cover_bytes = pix.tobytes("jpeg")
    except Exception as e:
        print(f"PDF cover extraction warning: {e}")

    text_content = [page.get_text() for page in doc]
    return ' '.join(text_content), title, author, cover_bytes


def read_mobi(file_path):
    tempdir, ext_path = mobi.extract(file_path)
    cover_bytes = None
    if ext_path.lower().endswith('.epub'):
        text, title, author, cover_bytes = read_epub(ext_path)
    else:
        with open(ext_path, 'r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
            title, author = None, None
    shutil.rmtree(tempdir, ignore_errors=True)
    return text, title, author, cover_bytes


def extract_text_and_metadata(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    raw_text = ""
    cover_bytes = None
    t, a = None, None

    if ext == '.epub':
        raw_text, t, a, cover_bytes = read_epub(file_path)
    elif ext == '.pdf':
        raw_text, t, a, cover_bytes = read_pdf(file_path)
    elif ext in ['.mobi', '.azw3']:
        raw_text, t, a, cover_bytes = read_mobi(file_path)
    elif ext == '.txt':
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            raw_text = f.read()
    elif ext in ['.htm', '.html']:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            raw_text = soup.get_text(separator=' ', strip=True)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    if t and t.strip() and t.strip().lower() != "untitled":
        title = t.strip()
    else:
        title = base_name.replace("_", " ").title()

    if a and a.strip() and a.strip().lower() != "untitled":
        author = a.strip()
    else:
        author = "Unknown Author"

    if not raw_text.strip():
        raise ValueError("No text could be extracted from the file.")
    text = clean_and_normalize_text(raw_text)

    return text, title, author, cover_bytes


def sanitize_filename(filename):
    if not filename:
        return "unknown_file"
    sanitized = re.sub(r'[\/*?:"<>|\']', "", filename)
    sanitized = sanitized.replace(" ", "_")
    sanitized = sanitized.strip(". ")
    return sanitized[:100]


def ensure_directory(directory_path):
    os.makedirs(directory_path, exist_ok=True)


def show_converted_audiobooks():
    output_dir = os.path.abspath(os.path.join("Working_files", "Book"))
    if not os.path.exists(output_dir):
        return []
    files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(('.mp3', '.m4b'))]
    return files if files else []


def build_atempo_filter(speed_val: float) -> str:
    """Build FFmpeg atempo filter chain for speeds between 0.3x and 2.0x."""
    speed_val = float(speed_val)
    if abs(speed_val - 1.0) < 1e-3:
        return ""
    filters = []
    remaining = speed_val
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    filters.append(f"atempo={remaining:.4f}")
    return ",".join(filters)


def prepare_clean_reference_wav(ref_audio_path: str, target_dir: str) -> str:
    """Convert reference audio (.mp3/.wav/.flac/etc.) to a clean 48kHz mono WAV file for VoxCPM 2."""
    ensure_directory(target_dir)
    out_wav = os.path.join(target_dir, "clean_reference_voice.wav")
    subprocess.run(
        [FFMPEG_EXE, "-y", "-i", ref_audio_path, "-ac", "1", "-ar", "48000", out_wav],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return out_wav


def polish_chunk_audio(wave_chunk: np.ndarray, sr: int, pause_ms: int = 200, fade_ms: int = 10) -> np.ndarray:
    """
    Apply a subtle 10ms cosine micro-fade at chunk edges to eliminate DC-offset clicks,
    and append a natural breath pause between chunks.
    """
    audio = np.asarray(wave_chunk, dtype=np.float32).flatten()
    if audio.size == 0:
        return audio

    fade_samples = int((fade_ms / 1000.0) * sr)
    if fade_samples > 0 and audio.size > fade_samples * 2:
        fade_in = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, fade_samples, dtype=np.float32)))
        fade_out = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, fade_samples, dtype=np.float32)))
        audio = audio.copy()
        audio[:fade_samples] *= fade_in
        audio[-fade_samples:] *= fade_out

    if pause_ms and int(pause_ms) > 0:
        silence_samples = int((int(pause_ms) / 1000.0) * sr)
        if silence_samples > 0:
            silence = np.zeros(silence_samples, dtype=np.float32)
            audio = np.concatenate([audio, silence])

    return audio


try:
    _GENERATE_SUPPORTS_SEED = "seed" in inspect.signature(model._generate).parameters
except Exception:
    _GENERATE_SUPPORTS_SEED = False


def run_voxcpm_generate(
    text: str,
    reference_wav_path=None,
    prompt_wav_path=None,
    prompt_text=None,
    cfg_value: float = 2.0,
    inference_timesteps: int = 22,
    normalize: bool = False,
    retry_badcase: bool = True,
    retry_badcase_max_times: int = 5,
    retry_badcase_ratio_threshold: float = 5.5,
    seed: int = 42,
):
    if seed is not None and int(seed) >= 0:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    kwargs = dict(
        text=text,
        reference_wav_path=reference_wav_path,
        prompt_wav_path=prompt_wav_path,
        prompt_text=prompt_text,
        cfg_value=float(cfg_value),
        inference_timesteps=int(inference_timesteps),
        normalize=bool(normalize),
        denoise=False,
        retry_badcase=bool(retry_badcase),
        retry_badcase_max_times=int(retry_badcase_max_times),
        retry_badcase_ratio_threshold=float(retry_badcase_ratio_threshold),
    )
    if _GENERATE_SUPPORTS_SEED and seed is not None and int(seed) >= 0:
        kwargs["seed"] = int(seed)

    with torch.no_grad():
        return model.generate(**kwargs)


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REF_AUDIO_PATH = (
    os.path.join(_SCRIPT_DIR, "default_voice.mp3")
    if os.path.exists(os.path.join(_SCRIPT_DIR, "default_voice.mp3"))
    else "default_voice.mp3"
)
DEFAULT_REF_TEXT = (
    "The birch canoe slid on the smooth planks. "
    "Glue the sheet to the dark blue background. "
    "It's easy to tell the depth of a well. "
    "The juice of lemons makes fine punch."
)


def apply_quality_preset(preset_name: str):
    """Update sliders automatically when the user switches the Quality vs. Speed preset."""
    if "Maximum Studio Quality" in preset_name:
        return (
            gr.update(value=22),     # num_steps_slider (LocDiT steps)
            gr.update(value=2.0),    # cfg_slider
            gr.update(value=200),    # max_phrase_slider
            gr.update(value=400),    # max_chunk_slider
            gr.update(value=5),      # retry_max_times_slider
            gr.update(value=5.2),    # retry_threshold_slider
            gr.update(value=200),    # chunk_pause_slider (ms)
            gr.update(value="320k"), # mp3_bitrate_dropdown
        )
    elif "Balanced" in preset_name:
        return (
            gr.update(value=14),
            gr.update(value=2.0),
            gr.update(value=250),
            gr.update(value=500),
            gr.update(value=3),
            gr.update(value=5.8),
            gr.update(value=180),
            gr.update(value="256k"),
        )
    else:  # Fast Draft
        return (
            gr.update(value=8),
            gr.update(value=2.0),
            gr.update(value=300),
            gr.update(value=650),
            gr.update(value=2),
            gr.update(value=6.5),
            gr.update(value=150),
            gr.update(value="192k"),
        )


def basic_tts(
    ref_audio_input,
    control_instruction_input,
    use_ultimate_cloning,
    ref_text_input,
    gen_file_input,
    speed,
    max_phrase_length,
    max_chunk_length,
    num_steps,
    cfg,
    retry_max_times,
    retry_threshold,
    chunk_pause_ms,
    mp3_bitrate,
    seed_input,
    wetext_normalize,
    progress=gr.Progress(),
):
    if not gen_file_input:
        raise gr.Error("Please upload at least one eBook file to generate audio!")

    try:
        processed_audiobooks = []
        num_ebooks = len(gen_file_input)
        ebook_frac = {"extract_text": 0.05, "infer": 0.90, "mp3_meta": 0.05}

        MAX_PHRASE_LENGTH = int(max_phrase_length)
        MAX_CHUNK_LENGTH_CHARS = int(max_chunk_length)
        NUM_GENERATION_STEPS = int(num_steps)
        CFG_SCALE = float(cfg)
        RETRY_MAX_TIMES = int(retry_max_times)
        RETRY_THRESHOLD = float(retry_threshold)
        PAUSE_MS = int(chunk_pause_ms)
        BITRATE = str(mp3_bitrate or "320k")
        SEED_VAL = int(seed_input) if seed_input is not None else 42

        clean_control = re.sub(r"[()（）]", "", (control_instruction_input or "")).strip()
        clean_ref_text = (ref_text_input or "").strip()

        for idx, ebook_file_data in enumerate(gen_file_input):
            current_ebook_base_progress = idx / float(num_ebooks)
            progress_offset_within_ebook = 0.0

            original_ebook_path = (
                ebook_file_data
                if isinstance(ebook_file_data, str)
                else getattr(ebook_file_data, "name", str(ebook_file_data))
            )

            if not os.path.exists(original_ebook_path):
                continue

            progress(current_ebook_base_progress, desc=f"Ebook {idx+1}/{num_ebooks}: Extracting text...")
            try:
                gen_text, ebook_title, ebook_author, cover_bytes = extract_text_and_metadata(original_ebook_path)
            except Exception as e:
                print(f"Extraction error: {e}")
                continue

            progress_offset_within_ebook += ebook_frac["extract_text"]
            overall_infer_start_frac = current_ebook_base_progress + (progress_offset_within_ebook / num_ebooks)

            run_uuid = uuid.uuid4().hex[:8]
            temp_chunks_dir = os.path.abspath(os.path.join(
                "Working_files", "temp_audio_chunks",
                f"{sanitize_filename(ebook_title)}_{run_uuid}"
            ))
            ensure_directory(temp_chunks_dir)
            chunk_file_paths = []

            initial_sentences = sent_tokenize(gen_text)
            intermediate_phrases = []
            for sentence in initial_sentences:
                if len(sentence) <= MAX_PHRASE_LENGTH:
                    intermediate_phrases.append(sentence)
                else:
                    current_part = sentence
                    while len(current_part) > MAX_PHRASE_LENGTH:
                        split_pos = -1
                        delimiters = [',', ';', '—', '–']
                        for delimiter in delimiters:
                            pos = current_part.rfind(delimiter, 0, MAX_PHRASE_LENGTH)
                            if pos > split_pos:
                                split_pos = pos
                        if split_pos == -1:
                            split_pos = current_part.rfind(' ', 0, MAX_PHRASE_LENGTH)
                        if split_pos == -1:
                            split_pos = MAX_PHRASE_LENGTH
                        intermediate_phrases.append(current_part[:split_pos+1].strip())
                        current_part = current_part[split_pos+1:].strip()
                    if current_part:
                        intermediate_phrases.append(current_part)

            text_super_chunks = []
            current_chunk = ""
            for phrase in intermediate_phrases:
                if len(current_chunk) + len(phrase) + 1 > MAX_CHUNK_LENGTH_CHARS:
                    if current_chunk:
                        text_super_chunks.append(current_chunk)
                    current_chunk = phrase
                else:
                    if current_chunk:
                        current_chunk += " " + phrase
                    else:
                        current_chunk = phrase
            if current_chunk:
                text_super_chunks.append(current_chunk)

            # Merge tiny trailing chunk (< 45 chars) to prevent short-input artifacts
            if len(text_super_chunks) > 1 and len(text_super_chunks[-1]) < 45:
                text_super_chunks[-2] = text_super_chunks[-2] + " " + text_super_chunks[-1]
                text_super_chunks.pop()

            del gen_text
            gc.collect()

            num_super_chunks = len(text_super_chunks)
            if num_super_chunks == 0:
                continue

            try:
                active_ref_wav = None
                active_prompt_wav = None
                active_prompt_text = None

                if ref_audio_input and os.path.exists(str(ref_audio_input)):
                    progress(overall_infer_start_frac, desc="Preparing 48kHz reference audio for VoxCPM 2...")
                    active_ref_wav = prepare_clean_reference_wav(str(ref_audio_input), temp_chunks_dir)
                    # Ultimate Hi-Fi Cloning is used when enabled, a transcript is present, and no style override is requested
                    if use_ultimate_cloning and clean_ref_text and not clean_control:
                        active_prompt_wav = active_ref_wav
                        active_prompt_text = clean_ref_text
                else:
                    progress(overall_infer_start_frac, desc="Generating Voice Design anchor clip for consistent book narration...")
                    design_desc = clean_control or "Calm, clear, expressive audiobook narrator"
                    anchor_text = f"({design_desc})This is a reference narration sample to keep the voice consistent throughout the entire audiobook."
                    anchor_wav = run_voxcpm_generate(
                        text=anchor_text,
                        cfg_value=CFG_SCALE,
                        inference_timesteps=NUM_GENERATION_STEPS,
                        normalize=wetext_normalize,
                        retry_badcase=True,
                        retry_badcase_max_times=RETRY_MAX_TIMES,
                        retry_badcase_ratio_threshold=RETRY_THRESHOLD,
                        seed=SEED_VAL,
                    )
                    anchor_data = anchor_wav[0] if isinstance(anchor_wav, (list, tuple)) else anchor_wav
                    if hasattr(anchor_data, "cpu"):
                        anchor_data = anchor_data.cpu().numpy()
                    anchor_path = os.path.join(temp_chunks_dir, "voice_design_anchor.wav")
                    sf.write(anchor_path, np.asarray(anchor_data, dtype=np.float32).flatten(), sampling_rate)
                    active_ref_wav = anchor_path

                ebook_start_time = time.time()
                progress_updater = EbookProgressUpdater(progress, num_super_chunks, idx, num_ebooks, ebook_start_time)

                for i, text_chunk in enumerate(text_super_chunks):
                    progress_updater.set_chunk_index(i)
                    print(f"\n[DEBUG] Starting Chunk {i+1}/{num_super_chunks} (Steps={NUM_GENERATION_STEPS}, CFG={CFG_SCALE})")
                    print(f"[DEBUG] Text Snippet: {text_chunk[:100]}...")
                    chunk_progress_start = overall_infer_start_frac + (i / num_super_chunks) * (ebook_frac["infer"] / num_ebooks)
                    progress_updater(chunk_progress_start)

                    if clean_control and not active_prompt_text:
                        formatted_chunk = f"({clean_control}){text_chunk}"
                    else:
                        formatted_chunk = text_chunk

                    try:
                        audio_tensor = run_voxcpm_generate(
                            text=formatted_chunk,
                            reference_wav_path=active_ref_wav,
                            prompt_wav_path=active_prompt_wav,
                            prompt_text=active_prompt_text,
                            cfg_value=CFG_SCALE,
                            inference_timesteps=NUM_GENERATION_STEPS,
                            normalize=wetext_normalize,
                            retry_badcase=RETRY_MAX_TIMES > 0,
                            retry_badcase_max_times=RETRY_MAX_TIMES,
                            retry_badcase_ratio_threshold=RETRY_THRESHOLD,
                            seed=SEED_VAL,
                        )
                        wave_data = audio_tensor[0] if isinstance(audio_tensor, (list, tuple)) else audio_tensor
                        if hasattr(wave_data, "cpu"):
                            wave_data = wave_data.cpu().numpy()

                        wave_chunk = np.asarray(wave_data, dtype=np.float32).flatten()
                        if wave_chunk is not None and wave_chunk.size > 0 and wave_chunk.any():
                            polished_chunk = polish_chunk_audio(wave_chunk, sampling_rate, pause_ms=PAUSE_MS, fade_ms=10)
                            chunk_path = os.path.join(temp_chunks_dir, f"chunk_{i:04d}.wav")
                            sf.write(chunk_path, polished_chunk, sampling_rate)
                            chunk_file_paths.append(chunk_path)
                    finally:
                        if 'audio_tensor' in locals():
                            del audio_tensor
                        if 'wave_chunk' in locals():
                            del wave_chunk
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                if not chunk_file_paths:
                    raise RuntimeError(f"No audio chunks were generated for '{ebook_title}'.")

                progress_offset_within_ebook += ebook_frac["infer"]
                progress(
                    current_ebook_base_progress + (progress_offset_within_ebook / num_ebooks),
                    desc=f"Ebook {idx+1}/{num_ebooks}: Mastering 48kHz ({BITRATE}) MP3...",
                )

                sanitized_title = sanitize_filename(ebook_title) or f"audiobook_{idx}"
                final_mp3_dir = os.path.abspath(os.path.join("Working_files", "Book"))
                final_mp3_path = os.path.join(final_mp3_dir, f"{sanitized_title}.mp3")

                concat_list_path = os.path.join(temp_chunks_dir, "concat_list.txt")
                with open(concat_list_path, 'w', encoding='utf-8') as f:
                    for path in chunk_file_paths:
                        f.write(f"file '{os.path.basename(path)}'\n")

                ensure_directory(final_mp3_dir)

                ffmpeg_command = [FFMPEG_EXE, '-f', 'concat', '-safe', '0', '-i', 'concat_list.txt']

                has_cover = False
                cover_path = os.path.join(temp_chunks_dir, "cover.jpg")
                if cover_bytes:
                    try:
                        with open(cover_path, "wb") as f:
                            f.write(cover_bytes)
                        has_cover = True
                    except Exception as e:
                        print(f"Failed to write cover image: {e}")

                if has_cover:
                    ffmpeg_command.extend([
                        '-i', 'cover.jpg',
                        '-map', '0:a',
                        '-map', '1:v',
                        '-c:v', 'mjpeg',
                        '-disposition:v', 'attached_pic'
                    ])

                atempo_filter = build_atempo_filter(speed)
                if atempo_filter:
                    ffmpeg_command.extend(['-filter:a', atempo_filter])

                ffmpeg_command.extend([
                    '-c:a', 'libmp3lame',
                    '-b:a', BITRATE,
                    '-ar', str(sampling_rate),
                    '-id3v2_version', '3',
                    '-metadata', f'title={ebook_title}',
                    '-metadata', f'artist={ebook_author}',
                    '-metadata', f'album={ebook_title}',
                    '-y', os.path.abspath(final_mp3_path)
                ])

                subprocess.run(ffmpeg_command, cwd=temp_chunks_dir, check=True)

                processed_audiobooks.append(final_mp3_path)
                yield processed_audiobooks

            except Exception as e:
                print(f"Book processing / FFmpeg error: {e}")
                traceback.print_exc()
            finally:
                if os.path.exists(temp_chunks_dir):
                    shutil.rmtree(temp_chunks_dir, ignore_errors=True)

        progress(1.0, desc=f"All {num_ebooks} eBook(s) processing finished.")

    except Exception as e:
        print("\n" + "="*50)
        print("CRITICAL ERROR DURING VOXCPM 2 TTS PROCESSING")
        traceback.print_exc()
        print("="*50 + "\n")
        raise gr.Error(f"An error occurred: {str(e)}")


def _on_ref_audio_change(audio_path):
    """Automatically enable Ultimate Hi-Fi Cloning for default_voice.mp3, or prompt for transcript on custom clips."""
    if not audio_path:
        return gr.update(value="", interactive=True), gr.update(value=False)
    if os.path.basename(str(audio_path)) == "default_voice.mp3":
        return gr.update(value=DEFAULT_REF_TEXT, interactive=True), gr.update(value=True)
    return (
        gr.update(
            value="",
            placeholder="Optional: Paste exact transcript of your uploaded clip here to use Ultimate Hi-Fi Cloning...",
        ),
        gr.update(value=False),
    )


def create_gradio_app():
    with gr.Blocks() as app:
        gr.Markdown("# eBook to Audiobook with VoxCPM 2 (48kHz Studio Quality)")

        # --- INPUT SECTION ---
        with gr.Column():
            quality_preset = gr.Radio(
                choices=[
                    "Maximum Studio Quality (22 Steps, 320kbps 48kHz, Strict Auto-Retry)",
                    "Balanced High Quality (14 Steps, 256kbps 48kHz)",
                    "Fast Draft (8 Steps, 192kbps 48kHz)",
                ],
                value="Maximum Studio Quality (22 Steps, 320kbps 48kHz, Strict Auto-Retry)",
                label="Quality vs. Speed Preset",
            )
            ref_audio_input = gr.Audio(
                label="Reference Voice (Use Default Voice, Upload Clip, Record, or Clear for Voice Design)",
                type="filepath",
                value=DEFAULT_REF_AUDIO_PATH
            )
            control_instruction_input = gr.Textbox(
                label="Voice Style / Control Instruction (Optional — steers emotion/pace or designs a voice; leave blank for Ultimate Hi-Fi Cloning)",
                placeholder="e.g., Calm, warm audiobook narrator with steady pacing (leave blank for pure Hi-Fi cloning)",
                value="",
                lines=1
            )
            gen_file_input = gr.File(
                label="Upload eBooks (Batch Support: .epub, .mobi, .pdf, .txt, .html)",
                file_types=[".epub", ".mobi", ".pdf", ".txt", ".html"],
                file_count="multiple",
                type="filepath"
            )
            generate_btn = gr.Button("Start Processing", variant="primary")

        # --- OUTPUT SECTION ---
        with gr.Column():
            gr.Markdown("### Current Batch Progress")
            batch_output = gr.Files(label="Audiobooks Finishing in Current Batch")

            gr.Markdown("### Completed Audiobooks Library")
            library_output = gr.Files(label="All Saved Audiobooks in Output Folder")
            show_audiobooks_btn = gr.Button("Refresh / Show All Saved Audiobooks", variant="secondary")

        with gr.Accordion("Advanced Quality & Generation Settings", open=False):
            use_ultimate_cloning = gr.Checkbox(
                label="Enable Ultimate Hi-Fi Cloning Mode (Best vocal fidelity — uses Reference Transcript below when Style Instruction is blank)",
                value=True
            )
            ref_text_input = gr.Textbox(
                label="Reference Audio Transcript (Pre-filled for default_voice.mp3; paste transcript here if using Ultimate Hi-Fi Mode with your own voice)",
                lines=2,
                value=DEFAULT_REF_TEXT
            )
            num_steps_slider = gr.Slider(
                label="LocDiT Flow-Matching Quality Steps (inference_timesteps)",
                minimum=4,
                maximum=35,
                value=22,
                step=1,
                info="Higher = smoother, cleaner 48kHz vocals. 10 = Fast Default | 20-25 = Maximum Studio Quality."
            )
            cfg_slider = gr.Slider(
                label="CFG Scale (Guidance)",
                minimum=1.0,
                maximum=3.0,
                value=2.0,
                step=0.1,
                info="2.0 is optimal for fidelity. Avoid >2.5 on long audiobooks as it can make voices sound tense."
            )
            with gr.Row():
                retry_max_times_slider = gr.Slider(
                    label="Bad-Case Auto-Retry Attempts",
                    minimum=0,
                    maximum=10,
                    value=5,
                    step=1,
                    info="Automatically re-generates any chunk that babbles or stretches words."
                )
                retry_threshold_slider = gr.Slider(
                    label="Bad-Case Strictness Threshold (Lower = Stricter)",
                    minimum=3.5,
                    maximum=8.0,
                    value=5.2,
                    step=0.1,
                    info="Max allowed audio-to-text ratio before triggering a re-generation (5.0-5.5 recommended for studio quality)."
                )
            with gr.Row():
                chunk_pause_slider = gr.Slider(
                    label="Natural Breath Pause Between Chunks (ms)",
                    minimum=0,
                    maximum=600,
                    value=200,
                    step=25,
                    info="Adds a subtle pause + 10ms anti-click fade between chunks."
                )
                mp3_bitrate_dropdown = gr.Dropdown(
                    choices=["320k", "256k", "192k"],
                    value="320k",
                    label="MP3 Output Bitrate (at 48kHz)"
                )
            speed_slider = gr.Slider(
                label="Speech Speed",
                minimum=0.5,
                maximum=2.0,
                value=1.0,
                step=0.05
            )
            with gr.Row():
                max_phrase_slider = gr.Slider(
                    label="Max Phrase Length (Chars)",
                    minimum=100,
                    maximum=1000,
                    value=200,
                    step=25,
                    info="200 keeps sentences tight for natural prosody."
                )
                max_chunk_slider = gr.Slider(
                    label="Max Chunk Length (Chars)",
                    minimum=200,
                    maximum=1500,
                    value=400,
                    step=25,
                    info="350-450 prevents attention degradation and voice drift on long chapters."
                )
            with gr.Row():
                seed_input = gr.Number(
                    label="Seed (-1 for random)",
                    value=42,
                    precision=0
                )
                wetext_normalize = gr.Checkbox(
                    label="Enable VoxCPM Built-in Text Normalizer (wetext)",
                    value=False,
                    info="Keep off for English eBooks (already normalized); enable for Chinese/multilingual books."
                )

        quality_preset.change(
            apply_quality_preset,
            inputs=[quality_preset],
            outputs=[
                num_steps_slider,
                cfg_slider,
                max_phrase_slider,
                max_chunk_slider,
                retry_max_times_slider,
                retry_threshold_slider,
                chunk_pause_slider,
                mp3_bitrate_dropdown,
            ],
            queue=False,
        )

        ref_audio_input.change(
            _on_ref_audio_change,
            inputs=[ref_audio_input],
            outputs=[ref_text_input, use_ultimate_cloning],
            queue=False
        )

        generate_btn.click(
            basic_tts,
            inputs=[
                ref_audio_input,
                control_instruction_input,
                use_ultimate_cloning,
                ref_text_input,
                gen_file_input,
                speed_slider,
                max_phrase_slider,
                max_chunk_slider,
                num_steps_slider,
                cfg_slider,
                retry_max_times_slider,
                retry_threshold_slider,
                chunk_pause_slider,
                mp3_bitrate_dropdown,
                seed_input,
                wetext_normalize,
            ],
            outputs=[batch_output]
        ).then(
            show_converted_audiobooks,
            inputs=[],
            outputs=[library_output],
            queue=False
        )

        show_audiobooks_btn.click(
            show_converted_audiobooks,
            inputs=[],
            outputs=[library_output],
            queue=False
        )

        app.load(show_converted_audiobooks, inputs=[], outputs=[library_output], queue=False)

    return app


if __name__ == "__main__":
    app = create_gradio_app()
    host = os.environ.get("VOXCPM_HOST", os.environ.get("OMNIVOICE_HOST", "127.0.0.1"))
    port = int(os.environ.get("VOXCPM_PORT", os.environ.get("OMNIVOICE_PORT", 7860)))

    app.queue(default_concurrency_limit=1).launch(
        theme=gr.themes.Ocean(),
        server_name=host,
        server_port=port,
        max_threads=10,
        show_error=True,
    )
