import openai

from config import OPENAI_API_KEY, EMBEDDING_MODEL, GPT_MODEL
from prompt import PROMPT_TEMPLATE

client = openai.OpenAI(api_key=OPENAI_API_KEY)

def get_embedding(texts: list[str]) -> list[float]:
	"""
  주어진 텍스트 리스트를 OpenAI Embedding API에 요청해 임베딩합니다.
	"""
	return openai.embeddings.create(input=texts, model=EMBEDDING_MODEL)

def generate_answer(question: str, context_chunks: list[str]) -> str:
  """
	유저의 질문과 관련 텍스트 청크를 OpenAI Chat API에 요청해 답변을 생성합니다.
  prompt.py에서 프롬프트 템플릿을 가져와 텍스트 청크를 문맥으로 제공합니다.
  """
  context = "\n\n".join(context_chunks)
  system_prompt = PROMPT_TEMPLATE.format(context=context)
  user_prompt = f"[question]\n{question}"

  response = client.chat.completions.create(
    model=GPT_MODEL,
    messages=[
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": user_prompt},
    ],
    temperature=0
  )
  return response.choices[0].message.content