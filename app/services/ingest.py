import fitz

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""

    for page in doc:
        text += page.get_text()

    return text


def chunk_text(text, chunk_size=1200, overlap=300):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]   # ✅ define chunk

        # 🔥 FILTER BAD CHUNKS
        if len(chunk.strip()) > 50:
            if "disclaimer" not in chunk.lower():
                if "credits" not in chunk.lower():
                    chunks.append(chunk)   # ✅ append only once

        start += chunk_size - overlap

    return chunks
