import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

from PyPDF2 import PdfReader
from docx import Document
from nltk.tokenize import sent_tokenize

def stream_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        # Read in blocks so it's faster than line by line, but keeps memory low
        while True:
            block = f.read(1024 * 1024) # 1MB blocks
            if not block:
                break
            yield block

def stream_pdf(path):
    reader = PdfReader(path)
    for page in reader.pages:
        text = page.extract_text()
        if text and text.strip():
            yield text

def stream_docx(path):
    doc = Document(path)
    for p in doc.paragraphs:
        if p.text.strip():
            yield p.text

def chunk_text_stream(text_stream, sentences_per_chunk=5, overlap=2):
    """
    Given an iterator of text blocks, yield chunks of text incrementally.
    """
    buffer = ""
    sentences = []
    
    for text_part in text_stream:
        buffer += " " + text_part
        new_sentences = sent_tokenize(buffer)
        
        # Keep the last sentence in the buffer because it might be incomplete
        if len(new_sentences) > 1:
            sentences.extend(new_sentences[:-1])
            buffer = new_sentences[-1]
        
        # Yield chunks if we have enough sentences
        while len(sentences) >= sentences_per_chunk:
            chunk = " ".join(sentences[:sentences_per_chunk])
            yield chunk.strip()
            # Slide window
            sentences = sentences[sentences_per_chunk - overlap:]
            
    # Process remaining buffer
    if buffer.strip():
        sentences.extend(sent_tokenize(buffer))
        
    while len(sentences) > 0:
        chunk = " ".join(sentences[:sentences_per_chunk])
        yield chunk.strip()
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

