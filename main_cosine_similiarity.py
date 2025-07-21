import os
import pickle

import numpy as np
import openai

from dotenv import load_dotenv
from utils import parse_pdf, chunk_text
from prompt import PROMPT_TEMPLATE


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

PDF_FILE_PATH = "data/whitepaper_Foundational Large Language models & text gen.pdf"
VECTOR_STORE_PATH = "index/vector_store.pkl"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-4o-mini"

document_store = []

def get_embedding(text: str) -> list[float]:
	response = openai.embeddings.create(input=[text], model=EMBEDDING_MODEL)
	return response.data[0].embedding

def cosine_similarity(v1, v2):
  	return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def setup_document_store():
	global document_store
	if os.path.exists(VECTOR_STORE_PATH):
		print(f"--- 기존 벡터 스토어 로딩: {VECTOR_STORE_PATH} ---")
		with open(VECTOR_STORE_PATH, "rb") as f:
			document_store = pickle.load(f)
		print("--- 벡터 스토어 로딩 완료 ---")	
		return

	print("--- 1. 문서 처리 시작 (신규 생성) ---")
	if not os.path.exists(PDF_FILE_PATH):
		print(f"오류: '{PDF_FILE_PATH}' 파일을 찾을 수 없습니다.")
		return

	raw_text = parse_pdf(PDF_FILE_PATH)
	print(f"문서 파싱 완료. 총 글자 수: {len(raw_text)}")

	chunks = chunk_text(raw_text, CHUNK_SIZE, CHUNK_OVERLAP)
	print(f"{len(chunks)}개의 텍스트 조각으로 분할 완료")

	print("임베딩 생성 및 저장 중...")
	temp_store = []
	for i, chunk in enumerate(chunks):
		embedding = get_embedding(chunk)
		temp_store.append({
			"text": chunk,
			"embedding": np.array(embedding)
		})
		print(f"  - {i+1}/{len(chunks)} 번째 조각 임베딩 완료")

	document_store = temp_store

	print(f"\n--- 생성된 벡터 스토어를 파일에 저장: {VECTOR_STORE_PATH} ---")
	os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
	with open(VECTOR_STORE_PATH, "wb") as f:
		pickle.dump(document_store, f)

	print("--- 문서 처리 및 저장 완료 ---")

def find_relevant_chunks(question: str, top_k: int = 5) -> list[str]:
	print("\n--- 2. 관련 문서 조각 검색 ---")

	question_embedding = np.array(get_embedding(question))

	similarities = [
		cosine_similarity(question_embedding, item["embedding"])
		for item in document_store
	]

	top_indices = np.argsort(similarities)[-top_k:][::-1]

	print(f"가장 관련성 높은 {top_k}개의 조각을 찾았습니다.")

	return [document_store[i]["text"] for i in top_indices]

def answer_with_gpt(question: str, relevant_chunks: list[str]) -> str:
	print(f"\n--- 3. {GPT_MODEL}를 통해 답변 생성 ---")

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
		model=GPT_MODEL,
		messages=messages,
		temperature=0.7
		)
		return response.choices[0].message.content
	except Exception as e:
		return f"API 호출 중 오류 발생: {e}"

if __name__ == "__main__":
	setup_document_store()

	if document_store:
		while True:
			user_question = input("\n질문을 입력하세요 (종료하려면 'exit' 입력):")
			if user_question.lower() == "exit":
				break
			relevant_chunks = find_relevant_chunks(user_question)

			print("\n--- 검색된 관련 조각 (LLM에 전달될 내용) ---")
			for i, chunk in enumerate(relevant_chunks):
				print(f"--- [조각 {i+1}] ---")
				print(chunk)
				print("-" * 20)
						
			answer = answer_with_gpt(user_question, relevant_chunks)
			print("\n--- 최종 답변 ---")
			print(answer)