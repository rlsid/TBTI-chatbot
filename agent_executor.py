from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt.tool_executor import ToolExecutor
from langgraph.graph.message import add_messages
from langchain_core.language_models import LanguageModelLike

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
    
    # 함수 호출 도구 사용할 수 있는 모델 생성
    model_with_tools = model.bind_tools(tools)

    # 모델 호출하는 함수 정의
    def call_model(state: AgentState):
        response = model_with_tools.invoke(state['messages'])
        final_response = response.content
        
        # AI 답변을 json 형식으로 만들어 final_response에 저장 / 답변 history에 저장 
        return {"final_response" : str({"answer": final_response, "place": None}), "messages" : [response]}

    # 도구 호출 시 최종 답변 생성하는 함수 정의(함수 실행 결과를 final_response 변수에 담음)
    def respond_after_calling_tools(state: AgentState):
        final_response = state["messages"][-1].content
        return {"final_response": final_response}
    
    # Define the function that determines whether to continue or not
    def should_continue(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        
        # 함수 호출이 없으면 바로 사용자에게 리턴
        if not last_message.tool_calls:
            return "end"
        # 있으면 워크플로우 지속
        else:
            return "continue"

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
            "continue": "tools",
            "end": END,
        },
    )
    
    workflow.add_edge("tools", "respond")
    workflow.add_edge("respond", END)

    graph = workflow.compile(
        checkpointer=checkpointer
    )

    return graph