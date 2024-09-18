import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict

from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from openAI_api import llm
from access_milvusDB import MilvusDB
from available_functions import callable_tools

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "api_key"
os.environ["LANGCHAIN_PROJECT"] = "프로젝트이름"

app = FastAPI()

initial_system_prompt = "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."

memory = MemorySaver()

tools = callable_tools

# 에이전트 생성
agent = create_react_agent(
    llm, 
    tools, 
    state_modifier=initial_system_prompt, 
    checkpointer=memory
)

config = {"configurable": {"thread_id": "test-thread"}}

class QuestionRequest(BaseModel):
    question: str


class AiResponse(BaseModel):
    answer: str
    place: Optional[List[Dict]] = None
    
db = MilvusDB()


@app.post("/ask-ai/", response_model=AiResponse)
async def ask_ai(request: QuestionRequest):
    question = request.question  # JSON에서 question 필드 추출

    try:
        db.reconnect()
        # 에이전트 실행
        response = agent.invoke({"messages": [("human", f"{question}")]}, config)
        ai_answer = response["messages"][-1].content
        print(AiResponse(answer=ai_answer))
        return AiResponse(answer=ai_answer)
    except Exception as e:
        print("에러 발생: ", e)
        raise HTTPException(status_code=500, detail="AI 처리 중 오류 발생")
    finally:
        db.unconnect()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
