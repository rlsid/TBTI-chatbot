from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

GPT_MODEL = 'gpt-4o'
EMBEDDING_MODEL = 'text-embedding-3-small'
DIMENSION = 768

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

print("openAI API 모듈 로드")