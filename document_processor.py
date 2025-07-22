import os

import pymupdf

def parse_pdf(file_path: str) -> str:
  if not os.path.exists(file_path):
    print(f"Error: '{file_path}' File not found.")
    return
		
  doc = pymupdf.open(file_path)
  text = ""
  for page in doc:
    text += page.get_text()
  doc.close()
  return text

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 32) -> list[str]:
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
