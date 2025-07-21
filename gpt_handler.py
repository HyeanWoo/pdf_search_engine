import openai

from config import OPENAI_API_KEY, EMBEDDING_MODEL, GPT_MODEL
from prompt import PROMPT_TEMPLATE

client = openai.OpenAI(api_key=OPENAI_API_KEY)

def get_embedding(texts: list[str]) -> list[float]:
	return openai.embeddings.create(input=texts, model=EMBEDDING_MODEL)

def generate_answer(question: str, context_chunks: list[str]) -> str:
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