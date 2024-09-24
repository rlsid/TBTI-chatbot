import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict

from langgraph.checkpoint.memory import MemorySaver
from openAI_api import llm
from access_milvusDB import database
from available_functions import callable_tools
from agent_executor import create_my_agent

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "자신의 langchain key 사용"
os.environ["LANGCHAIN_PROJECT"] = "프로젝트 이름"

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

config = {"configurable": {"thread_id": "test-thread"}}    

db = database

@app.post("/ask-ai/", response_model=AiResponse)
async def ask_ai(request: QuestionRequest):
    question = request.question  # JSON에서 question 필드 추출

    try:
        db.reconnect()
        # 에이전트 실행
        response = agent.invoke({"messages": [("human", f"{question}")]}, config)['final_response']
        response = response.strip("'<>() ").replace('\'', '\"').replace('None', 'null')
        ai_answer = json.loads(response)
        print("ai_answer:", ai_answer)
        return ai_answer
    
    except Exception as e:
        print("에러 발생: ", e)
        raise HTTPException(status_code=500, detail="AI 처리 중 오류 발생")
    finally:
        db.unconnect()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)