import os

from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY 환경변수를 찾을 수 없습니다. .env 파일을 확인하세요.")

PDF_FILE_PATH = "data/whitepaper_Foundational Large Language models & text gen.pdf"
FAISS_INDEX_PATH = "cache/vector_store.faiss"
CHUNK_STORE_PATH = "cache/chunks.pkl"

EMBEDDING_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-4o-mini"