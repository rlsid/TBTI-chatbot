import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict

from langgraph.checkpoint.memory import MemorySaver
from openAI_api import llm
from access_milvusDB import database
from available_functions import callable_tools
from agent_executor import create_my_agent


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = "TBTI_test3"

class QuestionRequest(BaseModel):
    question: str


class AiResponse(BaseModel):
    answer: str
    place: Optional[List[Dict]] = None


app = FastAPI()

# 대화 기록 메모리 생성
memory = MemorySaver()

# 호출할 함수 리스트 가져오기
tools = callable_tools

# 에이전트 생성 
agent = create_my_agent(
    model=llm,
    tools=tools,
    checkpointer=memory
)

config = {"configurable": {"thread_id": "test-thread2"}}    

db = database

@app.post("/ask-ai/", response_model=AiResponse)
async def ask_ai(request: QuestionRequest):
    question = request.question  # JSON에서 question 필드 추출

    try:
        db.reconnect()
        
        system_prompt = """
        - You are a tour guide called 'TBTI'. Ask the user a short and clear question.
        - If users request new information other than previous information, you don't use previously known information
        ex. user: 'Tell me the new destination, not the information you told me.'
        
        - Just ask once what kind of trip the user wants.
        ex. Is there anything you want when you travel?

        - Only up to five locations will be notified.
        - If the user wants specific information about multiple locations, it will only tell you the information of the previously recommended location.
        ex. Can you tell me where Wi-Fi is available among the places you told me?
        """

        messages_list = [("system", f"{system_prompt}")] 
        messages_list.append(("human", f"{question}"))

        # 에이전트 실행
        response = agent.invoke({"messages": messages_list}, config)['final_response']
        print(response)

        return response
    
    except Exception as e:
        print("에러 발생: ", e)
        raise HTTPException(status_code=500, detail="AI 처리 중 오류 발생")
    finally:
        db.unconnect()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)