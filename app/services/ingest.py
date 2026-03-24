import fitz
import re

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    return pages  # return per-page, not one big blob

def chunk_text(pages, chunk_size=600, overlap=100):
    chunks = []
    
    for page_num, text in enumerate(pages):
        # Normalize whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        
        current_chunk = ""
        prev_tail = ""  # overlap from previous chunk
        
        for sentence in sentences:
            candidate = (prev_tail + " " + current_chunk + " " + sentence).strip()
            
            if len(candidate) < chunk_size:
                current_chunk = (current_chunk + " " + sentence).strip()
            else:
                if current_chunk:
                    chunks.append({
                        "text": current_chunk,
                        "page": page_num
                    })
                    # keep last ~overlap chars as context seed for next chunk
                    prev_tail = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = sentence
        
        if current_chunk and len(current_chunk) > 80:
            chunks.append({
                "text": current_chunk,
                "page": page_num
            })
    
    return chunks