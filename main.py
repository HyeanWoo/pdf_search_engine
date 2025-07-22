import os

from config import FAISS_INDEX_PATH, CHUNK_STORE_PATH, PDF_FILE_PATH
from vector_retriever import build_index, save_index, load_index, retrieve_relevant_chunks
from document_processor import parse_pdf, chunk_text
from gpt_handler import generate_answer

def main():
	faiss_index, chunk_texts = load_index(FAISS_INDEX_PATH, CHUNK_STORE_PATH)
	
	if not faiss_index:
		raw_text = parse_pdf(PDF_FILE_PATH)
		chunks = chunk_text(raw_text)

		faiss_index, chunk_texts = build_index(chunks)
		save_index(faiss_index, chunk_texts, FAISS_INDEX_PATH, CHUNK_STORE_PATH)
  
	while True:
		user_question = input("\n> Enter your question (or enter 'exit' to quit): ")
		if user_question.lower() == "exit":
			break

		relevant_chunks = retrieve_relevant_chunks(user_question, faiss_index, chunk_texts)
		answer = generate_answer(user_question, relevant_chunks)
		
		print("\n--- GPT Answer ---\n")
		print(answer)

if __name__ == "__main__":
  main()
  