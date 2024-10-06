from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
from openai import OpenAI

GPT_MODEL = 'gpt-4o-2024-08-06'
EMBEDDING_MODEL = 'upskyy/bge-m3-korean'

client = OpenAI()
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

try:
    llm = ChatOpenAI(
        model=GPT_MODEL
    )

except Exception as e:
        print("- OpenAI API(gpt model or embedding model) error -")
        print(f"Exception: {e}")

# chat completions function
def chat_completion_request(messages, tools=None, tool_choice=None, response_format=None, model=GPT_MODEL):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

# embedding function
def embedding(text):
    try:
        vector = embedding_model.encode(text)
        return vector
    
    except Exception as e:
        print('임베딩 벡터 생성 실패')
        print(f"Exception: {e}")
        return e

print("openAI API 모듈 로드")