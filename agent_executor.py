from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt.tool_executor import ToolExecutor
from langgraph.graph.message import add_messages
from langchain_core.language_models import LanguageModelLike

from criteria_of_answers import system_informations_of_functions
from openAI_api import chat_completion_request

from typing import (
    Optional, 
    TypedDict,
    Annotated, 
    Union, 
    Sequence
) 

# 노드에 전달되는 state
class AgentState(TypedDict):
    # 사용자에게 전달되는 최종 메시지
    final_response : str
    # 대화 history 전달
    messages : Annotated[list, add_messages]


def create_my_agent(
    model: LanguageModelLike,
    tools: Union[ToolExecutor, Sequence[BaseTool], ToolNode],
    checkpointer: Optional[BaseCheckpointSaver] = None
) -> CompiledGraph:
    
    # LLM 시스템 프롬프트 리스트 로드 - 작동되는 함수에 따라 시스템 프롬프트 내용 다름
    configuration_for_answers = system_informations_of_functions

    # 함수 호출 도구 사용할 수 있는 모델 생성
    model_with_tools = model.bind_tools(tools)

    # 사용할 AI 모델 로드
    def call_model(state: AgentState):
        response = model_with_tools.invoke(state['messages'])
        final_response = response.content
        
        # AI 답변을 json 형식으로 만들어 final_response에 저장 / 답변 history에 저장 
        return {"final_response" :  str({"answer": final_response, "place": None}), "messages" : [response]}

    # 도구 작동 후 함수 결과 LLM에게 최종 전달 후 답변 생성
    def respond_after_calling_tools(state: AgentState):
        # 작동된 마지막 도구 메시지 가져오기
        messages = state["messages"]
        last_tool_message = messages[-1]
        
        # 작동된 함수 이름 가져오기
        name_of_functions_called = last_tool_message.name
        print(name_of_functions_called)
        
        # 함수 리턴값 가져오기 = 검색 결과 가져오기
        reference = last_tool_message.content

        # 작동된 도구에 맞는 시스템 프롬프트 가져오기
        system_prompt = configuration_for_answers[name_of_functions_called]['system_prompt']
        
        # 결과 참고해서 LLM 답변 생성
        messages = [
            {"role":"system", "content": f"{system_prompt}"},
            {"role": "user", "content":f"{reference}"}
        ]
        
        llm_response = chat_completion_request(
            messages=messages,
            response_format={"type":"json_object"}
        ).choices[0].message.content
    
        return {"final_response": llm_response}
    
    # Define the function that determines whether to continue or not
    def should_continue(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        
        # 함수 호출이 없으면 바로 사용자에게 리턴
        if not last_message.tool_calls:
            return "pass"
        # 있으면 워크플로우 지속
        else:
            return "work"


    # 새로운 그래프 정의
    workflow = StateGraph(AgentState)

    # agent, tools 노드 생성
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("respond", respond_after_calling_tools)

    # Set the entrypoint as `agent`
    # This means that this node is the first one called
    workflow.set_entry_point("agent")

    # We now add a conditional edge
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "work": "tools",
            "pass": END,
        },
    )
    
    workflow.add_edge("tools", "respond")
    workflow.add_edge("respond", END)

    graph = workflow.compile(
        checkpointer=checkpointer
    )

    return graph