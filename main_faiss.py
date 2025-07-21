import os
import pickle

import numpy as np
import openai
import faiss

from utils import parse_pdf, chunk_text
from config import EMBEDDING_MODEL, FAISS_INDEX_PATH, CHUNK_STORE_PATH, PDF_FILE_PATH
from prompt import PROMPT_TEMPLATE

faiss_index = None
chunk_texts = []

def get_embedding(text: str) -> list[float]:
	response = openai.embeddings.create(input=[text], model=EMBEDDING_MODEL)
	return response.data[0].embedding

def setup_document_store_faiss():
	global faiss_index, chunk_texts
	
	if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CHUNK_STORE_PATH):
		faiss_index = faiss.read_index(FAISS_INDEX_PATH)
		with open(CHUNK_STORE_PATH, "rb") as f:
			chunk_texts = pickle.load(f)
		return

	print("--- 1. 문서 처리 및 FAISS 인덱스 생성 시작 ---")
	if not os.path.exists(PDF_FILE_PATH):
		print(f"오류: '{PDF_FILE_PATH}' 파일을 찾을 수 없습니다.")
		return

	raw_text = parse_pdf(PDF_FILE_PATH)
	chunk_texts = chunk_text(raw_text)
	print(f"{len(chunk_texts)}개의 텍스트 조각으로 분할 완료")

	print("임베딩 생성 및 저장 중...")
	response = openai.embeddings.create(input=chunk_texts, model=EMBEDDING_MODEL)
	embeddings = [res.embedding for res in response.data]
	embedding_matrix = np.array(embeddings).astype("float32")

	dimension = embedding_matrix.shape[1]
	faiss_index = faiss.IndexFlatL2(dimension)
	faiss_index.add(embedding_matrix)

	print(f"--- FAISS 인덱스 생성 완료 (인덱스에 {faiss_index.ntotal}개 벡터 추가) ---")
	
	os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
	faiss.write_index(faiss_index, FAISS_INDEX_PATH)
	with open(CHUNK_STORE_PATH, "wb") as f:
		pickle.dump(chunk_texts, f)
		
	print(f"--- 생성된 인덱스를 '{FAISS_INDEX_PATH}'에, 텍스트 조각을 '{CHUNK_STORE_PATH}'에 저장했습니다. ---")


def find_relevant_chunks(question: str, top_k: int = 5) -> list[str]:
	print("\n--- 2. 관련 문서 조각 검색 (FAISS) ---")
	
	question_embedding = np.array([get_embedding(question)]).astype("float32")

	distances, indices = faiss_index.search(question_embedding, top_k)

	print(f"가장 관련성 높은 {top_k}개의 조각을 찾았습니다.")

	return [chunk_texts[i] for i in indices[0]]

def answer_with_gpt(question: str, relevant_chunks: list[str]) -> str:
	print(f"\n--- 3. 답변 생성 ---")

	context = "\n\n".join(relevant_chunks)
	system_prompt = PROMPT_TEMPLATE.format(context=context)
	user_prompt = f"""
	[질문]
	{question}
	"""

	messages = [
		{"role": "system", "content": system_prompt},
		{"role": "user", "content": user_prompt},
	]

	try:
		response = openai.chat.completions.create(
      model="gpt-4.1-nano",
      messages=messages,
      temperature=0.0
		)
		return response.choices[0].message.content
	except Exception as e:
		return f"API 호출 중 오류 발생: {e}"

if __name__ == "__main__":
	setup_document_store_faiss()

	if faiss_index:
		while True:
			user_question = input("\n질문을 입력하세요 (종료하려면 'exit' 입력):")
			if user_question.lower() == "exit":
				break
			relevant_chunks = find_relevant_chunks(user_question)

			answer = answer_with_gpt(user_question, relevant_chunks)
			print("\n--- 최종 답변 ---")
			print(answer)