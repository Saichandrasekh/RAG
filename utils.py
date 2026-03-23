import os
import io
import subprocess
import tempfile
import logging
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

import fitz  # pymupdf
from docx import Document
from nltk.tokenize import sent_tokenize

# Optional OCR dependencies
try:
    from PIL import Image
    import pytesseract
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False

logger = logging.getLogger(__name__)

def stream_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            yield block, None

def stream_pdf(path):
    doc = fitz.open(path)
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text and text.strip():
            yield text, page_num
    doc.close()

def stream_docx(path):
    doc = Document(path)
    for p in doc.paragraphs:
        if p.text.strip():
            yield p.text, None

def extract_pdf_images(path, output_dir, min_size=50, max_ocr_dim=2000):
    """Extract embedded images from PDF, run OCR, save to output_dir.
    Yields dicts: {image_path, ocr_text, page, image_index, ext}.
    Skips tiny images (decorative elements) under min_size px."""
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(path)
    for page_num, page in enumerate(doc):
        for img_idx, img in enumerate(page.get_images(full=True)):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]
                ext = base_image["ext"]
                width = base_image.get("width", 0)
                height = base_image.get("height", 0)

                # Skip tiny decorative images
                if width < min_size or height < min_size:
                    continue

                img_path = os.path.join(output_dir, f"page_{page_num}_{img_idx}.{ext}")
                with open(img_path, "wb") as f:
                    f.write(img_bytes)

                # OCR the image
                ocr_text = ""
                if _OCR_AVAILABLE:
                    try:
                        pil_img = Image.open(io.BytesIO(img_bytes))
                        # Resize large images for memory safety
                        if max(pil_img.size) > max_ocr_dim:
                            pil_img.thumbnail((max_ocr_dim, max_ocr_dim))
                        ocr_text = pytesseract.image_to_string(pil_img).strip()
                    except Exception as e:
                        logger.warning(f"[OCR] Failed for page {page_num} img {img_idx}: {e}")

                if not ocr_text:
                    ocr_text = f"[Image on page {page_num}]"

                yield {
                    "image_path": img_path,
                    "ocr_text": ocr_text,
                    "page": page_num,
                    "image_index": img_idx,
                    "ext": ext,
                }
            except Exception as e:
                logger.warning(f"[IMAGE] Failed to extract page {page_num} img {img_idx}: {e}")
                continue
    doc.close()


def _table_to_markdown(rows):
    """Convert a list of rows (each a list of cell strings) to markdown table."""
    if not rows or len(rows) < 1:
        return ""
    # Clean cells: replace None with empty string, strip whitespace
    clean = []
    for row in rows:
        clean.append([str(cell).strip() if cell else "" for cell in row])

    # Build markdown: first row as header
    header = clean[0]
    md_lines = ["| " + " | ".join(header) + " |"]
    md_lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in clean[1:]:
        # Pad row if shorter than header
        padded = row + [""] * (len(header) - len(row))
        md_lines.append("| " + " | ".join(padded[:len(header)]) + " |")
    return "\n".join(md_lines)


def extract_pdf_tables(path):
    """Extract tables from PDF pages using PyMuPDF's find_tables().
    Yields (markdown_text, page_num, table_index) for each table found."""
    doc = fitz.open(path)
    for page_num, page in enumerate(doc):
        try:
            tables = page.find_tables()
        except AttributeError:
            # PyMuPDF version too old for find_tables()
            break
        for table_idx, table in enumerate(tables):
            try:
                rows = table.extract()
                md = _table_to_markdown(rows)
                if md.strip():
                    yield md, page_num, table_idx
            except Exception:
                continue
    doc.close()


def extract_docx_tables(path):
    """Extract tables from DOCX files using python-docx.
    Yields (markdown_text, None, table_index) for each table found."""
    doc = Document(path)
    for table_idx, table in enumerate(doc.tables):
        rows = []
        for row in table.rows:
            rows.append([cell.text.strip() for cell in row.cells])
        md = _table_to_markdown(rows)
        if md.strip():
            yield md, None, table_idx

def chunk_text_stream(text_stream, sentences_per_chunk=5, overlap=2):
    buffer = ""
    sentences = []  # list of (sentence_text, page_num)
    current_page = None

    for text_part, page_num in text_stream:
        if page_num is not None:
            current_page = page_num

        buffer += " " + text_part
        new_sentences = sent_tokenize(buffer)

        if len(new_sentences) > 1:
            sentences.extend([(s, current_page) for s in new_sentences[:-1]])
            buffer = new_sentences[-1]

        while len(sentences) >= sentences_per_chunk:
            chunk_sents = sentences[:sentences_per_chunk]
            chunk = " ".join(s for s, _ in chunk_sents)
            page = chunk_sents[0][1]
            yield chunk.strip(), page
            sentences = sentences[sentences_per_chunk - overlap:]

    if buffer.strip():
        sentences.extend([(s, current_page) for s in sent_tokenize(buffer)])

    while len(sentences) > 0:
        chunk_sents = sentences[:sentences_per_chunk]
        chunk = " ".join(s for s, _ in chunk_sents)
        page = chunk_sents[0][1]
        yield chunk.strip(), page
        if len(sentences) <= sentences_per_chunk:
            break
        sentences = sentences[sentences_per_chunk - overlap:]

def yield_file_chunks(path, sentences_per_chunk=3, overlap=1):
    """Yield (text, page, metadata_dict) tuples for all content types.
    metadata_dict contains at minimum {"type": "text"|"table"|"image"}.
    """
    ext = path.lower()

    # ── Text chunks ──
    if ext.endswith(".txt"):
        stream = stream_txt(path)
    elif ext.endswith(".pdf"):
        stream = stream_pdf(path)
    elif ext.endswith(".docx"):
        stream = stream_docx(path)
    else:
        return

    for chunk, page in chunk_text_stream(stream, sentences_per_chunk, overlap):
        yield chunk, page, {"type": "text"}

    # ── Table chunks (PDF and DOCX only) ──
    if ext.endswith(".pdf"):
        for md_text, page_num, table_idx in extract_pdf_tables(path):
            yield md_text, page_num, {"type": "table", "table_index": table_idx}
    elif ext.endswith(".docx"):
        for md_text, page_num, table_idx in extract_docx_tables(path):
            yield md_text, page_num, {"type": "table", "table_index": table_idx}

    # ── Image chunks (PDF only) ──
    if ext.endswith(".pdf"):
        filename_stem = os.path.splitext(os.path.basename(path))[0]
        images_dir = os.path.join("static", "images", filename_stem)
        for img_info in extract_pdf_images(path, images_dir):
            # Store web-relative path for serving
            web_path = f"/static/images/{filename_stem}/page_{img_info['page']}_{img_info['image_index']}.{img_info['ext']}"
            yield img_info["ocr_text"], img_info["page"], {
                "type": "image",
                "image_path": web_path,
                "image_index": img_info["image_index"],
            }


# ── Audio/Video helpers ──────────────────────────────────────────────────────

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a"}
VIDEO_EXTENSIONS = {".mp4", ".webm"}
AV_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


def is_audio_video(path):
    """Check if the file is an audio or video file."""
    return os.path.splitext(path.lower())[1] in AV_EXTENSIONS


def extract_audio(input_path, output_wav=None):
    """Extract audio from a video file or convert audio to WAV.
    Uses ffmpeg subprocess for reliability and low memory usage.
    Returns path to the WAV file."""
    ext = os.path.splitext(input_path.lower())[1]
    if output_wav is None:
        output_wav = tempfile.mktemp(suffix=".wav")

    if ext == ".wav":
        return input_path  # Already WAV, no conversion needed

    try:
        subprocess.run(
            ["ffmpeg", "-i", input_path, "-vn", "-acodec", "pcm_s16le",
             "-ar", "16000", "-ac", "1", "-y", output_wav],
            check=True, capture_output=True, timeout=300,
        )
        return output_wav
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning(f"[AUDIO] ffmpeg failed: {e}. Trying moviepy fallback...")
        try:
            from moviepy.editor import VideoFileClip, AudioFileClip
            if ext in VIDEO_EXTENSIONS:
                clip = VideoFileClip(input_path)
                clip.audio.write_audiofile(output_wav, codec="pcm_s16le",
                                           fps=16000, nbytes=2, logger=None)
                clip.close()
            else:
                clip = AudioFileClip(input_path)
                clip.write_audiofile(output_wav, codec="pcm_s16le",
                                     fps=16000, nbytes=2, logger=None)
                clip.close()
            return output_wav
        except Exception as e2:
            logger.error(f"[AUDIO] moviepy fallback also failed: {e2}")
            raise


def split_audio_for_whisper(wav_path, max_size_mb=24, segment_secs=600):
    """Split a WAV file into segments if it exceeds Whisper's 25MB limit.
    Returns list of file paths."""
    file_size = os.path.getsize(wav_path) / (1024 * 1024)
    if file_size <= max_size_mb:
        return [wav_path]

    # Split into segments using ffmpeg
    output_dir = tempfile.mkdtemp()
    pattern = os.path.join(output_dir, "segment_%03d.wav")
    try:
        subprocess.run(
            ["ffmpeg", "-i", wav_path, "-f", "segment",
             "-segment_time", str(segment_secs),
             "-c", "copy", "-y", pattern],
            check=True, capture_output=True, timeout=300,
        )
        segments = sorted([
            os.path.join(output_dir, f) for f in os.listdir(output_dir)
            if f.endswith(".wav")
        ])
        return segments if segments else [wav_path]
    except Exception as e:
        logger.warning(f"[AUDIO] Split failed: {e}. Using full file.")
        return [wav_path]


def _format_timestamp(seconds):
    """Convert seconds to MM:SS format."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def chunk_transcript(segments, sentences_per_chunk=5, overlap=2):
    """Chunk Whisper transcript segments into groups with timestamp ranges.
    segments: list of dicts with 'text', 'start', 'end' keys.
    Yields (chunk_text, timestamp_range) tuples."""
    if not segments:
        return

    # Flatten segments into sentences with timestamps
    sentence_data = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        sents = sent_tokenize(text)
        for sent in sents:
            sentence_data.append({
                "text": sent,
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
            })

    # Group into chunks with overlap
    i = 0
    while i < len(sentence_data):
        chunk_sents = sentence_data[i:i + sentences_per_chunk]
        if not chunk_sents:
            break
        chunk_text = " ".join(s["text"] for s in chunk_sents)
        start_ts = _format_timestamp(chunk_sents[0]["start"])
        end_ts = _format_timestamp(chunk_sents[-1]["end"])
        timestamp = f"{start_ts}-{end_ts}"
        yield chunk_text.strip(), timestamp

        if i + sentences_per_chunk >= len(sentence_data):
            break
        i += sentences_per_chunk - overlap
