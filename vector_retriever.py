import os
import pickle

import numpy as np
import faiss

from gpt_handler import get_embedding
from typing import Optional

def build_index(texts: list[str]) -> tuple[faiss.Index, list[str]]:
	"""
	주어진 텍스트 리스트를 임베딩하고 벡터화된 데이터를 인덱싱합니다.
  인덱싱은 faiss의 유클리드 거리 기반의 인덱스로 진행합니다.
  """
	response = get_embedding(texts)
	embeddings = [res.embedding for res in response.data]
	embedding_matrix = np.array(embeddings).astype("float32")

	faiss_index = faiss.IndexFlatL2(embedding_matrix.shape[1])
	faiss_index.add(embedding_matrix)

	return faiss_index, texts

def save_index(faiss_index: faiss.Index, chunk_texts: list[str], index_path: str, chunk_path: str):
	"""
	생성된 인덱스와 텍스트 청크를 로컬에 파일로 저장합니다.
	"""
	os.makedirs(os.path.dirname(index_path), exist_ok=True)
	faiss.write_index(faiss_index, index_path)

	with open(chunk_path, "wb") as f:
		pickle.dump(chunk_texts, f)

def load_index(index_path: str, chunk_path: str) -> tuple[Optional[faiss.Index], list[str]]:
	"""
	지정된 경로에서 인덱스 파일과 텍스트 청크 파일을 가져옵니다.
	"""
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
	"""
	주어진 유저의 질문을 임베딩하여 인덱스와 비교합니다.
	가장 연관성이 높은 텍스트 청크를 top_k개 만큼 가져와 반환합니다.
	"""
	response = get_embedding([question])
	question_embedding = np.array([response.data[0].embedding]).astype("float32")

	_, indices = faiss_index.search(question_embedding, top_k)

	return [chunk_texts[i] for i in indices[0]]