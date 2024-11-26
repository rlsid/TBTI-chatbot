import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt.tool_executor import ToolExecutor
from langgraph.prebuilt import ToolNode
from langchain_core.tools import BaseTool
from langchain_core.language_models import LanguageModelLike
from langgraph.graph.graph import CompiledGraph
from openAI_api import llm
from access_milvusDB import database
from callable_tools.helping_travel import tools_of_travel
from callable_tools.identifying_type import tools_of_type
from agent_executor import create_my_agent
from typing import (
    Optional,
    Union, 
    Sequence
) 

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "api key"
os.environ["LANGCHAIN_PROJECT"] = "test"

# 사용자별 메모리 저장용 딕셔너리
user_checkpointers = {}

# 사용자별 메모리 생성 함수
def get_user_checkpointer(userId: Optional[str]) -> MemorySaver:
    global user_checkpointers
    if not userId or userId == "false":
        return MemorySaver()
    if userId not in user_checkpointers:
        user_checkpointers[userId] = MemorySaver()
    return user_checkpointers[userId]

# 사용자별 에이전트 생성 함수
def create_user_agent(userId: Optional[str], model: LanguageModelLike, tools: Union[ToolExecutor, Sequence[BaseTool], ToolNode]) -> CompiledGraph:
    # 사용자별 mcheckpointer 가져오기
    user_checkpointer = get_user_checkpointer(userId)

    # 에이전트 생성
    return create_my_agent(
        model=model,
        tools=tools,
        checkpointer=user_checkpointer
    )

class QuestionRequest(BaseModel):
    userMessage: str
    userId: Optional[str] = None
    tbtiType: Optional[str] = None


class AiResponse(BaseModel):
    answer: str
    place: Optional[List[Dict]] = None
    

app = FastAPI()


# 호출할 함수 리스트 가져오기
tools = tools_of_travel["list_of_func"] + tools_of_type["list_of_func"]


# 이전 state 값 저장
previous_state = {
    "messages" : None,
    "previous_result" : None,
    "final_response" : None,
    "tbti_of_user" : None,
    "filtering" : None
}

db = database

@app.post("/ask-ai/", response_model=AiResponse)
async def ask_ai(request: QuestionRequest):
    global previous_state
    userId = request.userId or "false"
    userMessage = request.userMessage
    tbtiType = request.tbtiType

    try:
        db.reconnect()
        
        # 사용자 ID를 기반으로 에이전트 생성
        user_agent = create_user_agent(userId, llm, tools)
        
         # TBTI 유형 저장
        previous_state["tbti_of_user"] = tbtiType

        # 사용자 ID를 포함한 설정 구성
        config = {"configurable": {"thread_id": userId, "user_id": userId}}
        
        system_prompt = """
        - You are a tour guide called 'TBTI'. Ask the user a short and clear question.
        - Only up to five locations will be notified.
        - Don't ask a question what type of trip the user wants.
        """

        messages_list = [("system", f"{system_prompt}")] 
        messages_list.append(("human", f"{userMessage}"))
        previous_state["messages"] = messages_list

        # 에이전트 실행
        response = user_agent.invoke(previous_state, config)
        print(response)
        previous_state = response

        # JSON 직렬화 시 SecretStr 값 처리
        return response['final_response']
    
    except Exception as e:
        print("에러 발생: ", e)
        raise HTTPException(status_code=500, detail="AI 처리 중 오류 발생")
    finally:
        db.unconnect()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
