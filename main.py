import os

from config import FAISS_INDEX_PATH, CHUNK_STORE_PATH, PDF_FILE_PATH
from vector_retriever import build_index, save_index, load_index, retrieve_relevant_chunks
from document_processor import parse_pdf, chunk_text
from gpt_handler import generate_answer

def main():
	faiss_index, chunk_texts = load_index(FAISS_INDEX_PATH, CHUNK_STORE_PATH)
	
	if not faiss_index:
		if not os.path.exists(PDF_FILE_PATH):
			print(f"오류: '{PDF_FILE_PATH}' 파일을 찾을 수 없습니다.")
			return
		
		raw_text = parse_pdf(PDF_FILE_PATH)
		chunks = chunk_text(raw_text)
		faiss_index, chunk_texts = build_index(chunks)

		save_index(faiss_index, chunk_texts, FAISS_INDEX_PATH, CHUNK_STORE_PATH)
  
	while True:
		user_question = input("\n질문을 입력하세요 (종료하려면 'exit' 입력):")
		if user_question.lower() == "exit":
			break

		relevant_chunks = retrieve_relevant_chunks(
			question=user_question,
			faiss_index=faiss_index,
			chunk_texts=chunk_texts,
		)

		answer = generate_answer(
			question=user_question,
			context_chunks=relevant_chunks,
		)
		
		print("\n--- 최종 답변 ---")
		print(answer)

if __name__ == "__main__":
  main()
  