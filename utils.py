import pymupdf

def parse_pdf(file_path: str) -> str:
  doc = pymupdf.open(file_path)
  text = ""
  for page in doc:
    text += page.get_text()
  doc.close()
  return text

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
  if (chunk_size <= overlap):
    raise ValueError("chunk_size should bigger than overlap")
  
  chunks = []
  start = 0
  while start < len(text):
    end = start + chunk_size
    chunks.append(text[start:end])
    start += chunk_size - overlap
  return chunks
