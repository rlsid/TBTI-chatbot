import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict

from langgraph.checkpoint.memory import MemorySaver
from openAI_api import llm
from access_milvusDB import database
from callable_tools.helping_travel import tools_of_travel
from callable_tools.identifying_type import tools_of_type
from agent_executor import create_my_agent


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "key"
os.environ["LANGCHAIN_PROJECT"] = "test name"

class QuestionRequest(BaseModel):
    question: str


class AiResponse(BaseModel):
    answer: str
    place: Optional[List[Dict]] = None


app = FastAPI()

# 대화 기록 메모리 생성
memory = MemorySaver()

# 호출할 함수 리스트 가져오기
tools = tools_of_travel + tools_of_type["list_of_func"]

# 에이전트 생성 
agent = create_my_agent(
    model=llm,
    tools=tools,
    checkpointer=memory
)


config = {"configurable": {"thread_id": "test"}}    

# 이전 state 값 저장
previous_state = {
    "messages" : None,
    "previous_result" : None,
    "final_response" : None,
    "tbti_of_user" : "ASFU",
    "filtering" : None,
    "name_of_tools": None,
    "model_with_tools" : llm.bind_tools(tools)
}

# DB 가져오기
db = database

@app.post("/ask-ai/", response_model=AiResponse)
async def ask_ai(request: QuestionRequest):
    question = request.question  # JSON에서 question 필드 추출

    try:
        db.reconnect()
        
        system_prompt = """
        - You are a tour guide called 'TBTI'. Ask the user a short and clear question.
        - Only up to five locations will be notified.
        - If users request new information other than previous information, you don't use previously known information
        ex. user: 'Tell me the new destination, not the information you told me.'
        """

        messages_list = [("system", f"{system_prompt}")] 
        messages_list.append(("human", f"{question}"))
        previous_state["messages"] = messages_list

        # 에이전트 실행
        response = agent.invoke(previous_state, config)
        previous_state = response

        return response['final_response']
    
    except Exception as e:
        print("에러 발생: ", e)
        raise HTTPException(status_code=500, detail="AI 처리 중 오류 발생")
    finally:
        db.unconnect()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
