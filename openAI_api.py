from openai import OpenAI

GPT_MODEL = 'gpt-4o'
EMBEDDING_MODEL = 'text-embedding-3-small'
DIMENSION = 768

client = OpenAI()

# chat completions function
def chat_completion_request(messages, tools=None, tool_choice=None, model=GPT_MODEL):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e
    

# embedding function
def embedding(question, model=EMBEDDING_MODEL, dimention=DIMENSION):
    try:
        response = client.embeddings.create(
            input=question,
            model=model,
            dimensions=dimention
        )
        return response.data[0].embedding
    
    except Exception as e:
        print('임베딩 벡터 생성 실패')
        print(f"Exception: {e}")
        return e
