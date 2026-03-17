import os
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

def extract_pdf_images(path, output_dir):
    """Extract only embedded images (figures/diagrams) from each PDF page."""
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(path)
    for page_num, page in enumerate(doc):
        for img_idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            ext = base_image["ext"]
            img_path = os.path.join(output_dir, f"page_{page_num}_{img_idx}.{ext}")
            with open(img_path, "wb") as f:
                f.write(img_bytes)
    doc.close()

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

def yield_file_chunks(path, sentences_per_chunk=5, overlap=2):
    if path.lower().endswith(".txt"):
        stream = stream_txt(path)
    elif path.lower().endswith(".pdf"):
        stream = stream_pdf(path)
    elif path.lower().endswith(".docx"):
        stream = stream_docx(path)
    else:
        return

    yield from chunk_text_stream(stream, sentences_per_chunk, overlap)
