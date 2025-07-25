import os

import pymupdf

def parse_pdf(file_path: str) -> str:
  """
  file_path에 있는 PDF 문서를 확인 후 텍스트를 파싱하여 추출합니다.
  """
  if not os.path.exists(file_path):
    print(f"Error: '{file_path}' File not found.")
    return
		
  doc = pymupdf.open(file_path)
  text = ""
  for page in doc:
    text += page.get_text()
  doc.close()
  return text

def chunk_text(text: str, chunk_size: int = 256, overlap: int = 38) -> list[str]:
  """
  주어진 text를 특정 단위의 청크로 분리합니다.
  """
  if (chunk_size <= overlap):
    raise ValueError("chunk_size should bigger than overlap.")
  
  words = text.split()
  chunks = []
  start = 0
  offset = chunk_size - overlap

  while start < len(words):
    end = start + chunk_size
    chunk = " ".join(words[start:end])
    chunks.append(chunk)
    start += offset

  return chunks
