from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-base-en-v1.5')

def get_embedding(text: str):
    return model.encode(text, normalize_embeddings=True).tolist()

def get_embeddings_batch(texts: list):
    return model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=True).tolist()