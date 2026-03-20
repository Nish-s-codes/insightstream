from sentence_transformers import SentenceTransformer

# load model once
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text):
    emb = model.encode(text, normalize_embeddings=True)
    return emb.tolist()