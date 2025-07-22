import os
import pickle

import numpy as np
import faiss

from gpt_handler import get_embedding
from typing import Optional

def build_index(texts: list[str]) -> tuple[faiss.Index, list[str]]:
  response = get_embedding(texts)
  embeddings = [res.embedding for res in response.data]
  embedding_matrix = np.array(embeddings).astype("float32")

  faiss_index = faiss.IndexFlatL2(embedding_matrix.shape[1])
  faiss_index.add(embedding_matrix)

  return faiss_index, texts

def save_index(faiss_index: faiss.Index, chunk_texts: list[str], index_path: str, chunk_path: str):
  os.makedirs(os.path.dirname(index_path), exist_ok=True)
  faiss.write_index(faiss_index, index_path)

  with open(chunk_path, "wb") as f:
    pickle.dump(chunk_texts, f)

def load_index(index_path: str, chunk_path: str) -> tuple[Optional[faiss.Index], list[str]]:
  if os.path.exists(index_path) and os.path.exists(chunk_path):
    faiss_index = faiss.read_index(index_path)

    with open(chunk_path, "rb") as f:
      chunk_texts = pickle.load(f)

    return faiss_index, chunk_texts
		
  return None, None
  
def retrieve_relevant_chunks(
    question: str, 
    faiss_index: faiss.Index, 
    chunk_texts: list[str], 
    top_k: int = 5
  ) -> list[str]:
  response = get_embedding([question])
  question_embedding = np.array([response.data[0].embedding]).astype("float32")

  _, indices = faiss_index.search(question_embedding, top_k)

  return [chunk_texts[i] for i in indices[0]]