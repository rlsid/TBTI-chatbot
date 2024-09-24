from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

GPT_MODEL = 'gpt-4o'
EMBEDDING_MODEL = 'text-embedding-3-small'
DIMENSION = 768

client = OpenAI()

try:
    llm = ChatOpenAI(
        model=GPT_MODEL
    )

    embed =  OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        dimensions=DIMENSION
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
def embedding(question, model=EMBEDDING_MODEL, dimension=DIMENSION):
    try:
        response = client.embeddings.create(
            input=question,
            model=model,
            dimensions=dimension
        )
        return response.data[0].embedding
    
    except Exception as e:
        print('임베딩 벡터 생성 실패')
        print(f"Exception: {e}")
        return e

print("openAI API 모듈 로드")