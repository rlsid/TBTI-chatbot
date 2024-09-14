import os

from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.pydantic_v1 import BaseModel, Field

from openAI_api import llm
from access_milvusDB import db
from available_functions import callable_tools

# langsmith, langchain 환경 변수 설정
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "본인의 lanchain api key 입력"
os.environ["LANGCHAIN_PROJECT"] = "langchain 프로젝트 이름 입력"


def ask_ai(agent, config):
    while True:
        # 사용자 질문 받기 부분 - q라고 쓰면 대화 종료
        question = input("사용자 : ")

        if(question == 'q'):
            # DB 연결 끊기
            db.unconnect()
            break

        # 에이전트 실행
        response = agent.invoke({"messages": [("human", f"{question}")]}, config)
        ai_answer = response["messages"][-1].content
        print("ai: ", ai_answer)
        
        

# 대화의 초기 프롬프트 생성
initial_system_prompt = "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."

# 대화 기록 메모리 생성
memory = MemorySaver()

# 함수 호출 도구 준비
tools = callable_tools

# 함수 호출 에이전트 생성 - 호출 가능한 함수에 대한 정보를 전달
agent = create_react_agent(
    llm, 
    tools, 
    state_modifier=initial_system_prompt, 
    checkpointer=memory
)

config = {"configurable": {"thread_id": "test-thread"}}

# 대화 메인 실행
ask_ai(agent, config)

