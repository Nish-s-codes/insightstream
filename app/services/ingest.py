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
    
    current_chunk = ""
    current_page = 0
    
    for page_num, text in enumerate(pages):
        # Normalize whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        
        if not current_chunk:
            current_page = page_num
            
        for sentence in sentences:
            if not sentence:
                continue
                
            candidate = (current_chunk + " " + sentence).strip()
            
            if len(candidate) < chunk_size:
                current_chunk = candidate
            else:
                if current_chunk:
                    chunks.append({
                        "text": current_chunk,
                        "page": current_page
                    })
                    # Use last overlap chars directly in new current_chunk
                    prev_tail = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                    current_chunk = (prev_tail + " " + sentence).strip()
                else:
                    current_chunk = sentence
                current_page = page_num
        
    if current_chunk:
        chunks.append({
            "text": current_chunk,
            "page": current_page
        })
    
    return chunks