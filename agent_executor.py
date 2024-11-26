import json
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver 
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt.tool_executor import ToolExecutor
from langgraph.graph.message import add_messages

from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables.base import Runnable
from langchain_core.prompt_values import PromptValue
from langchain_core.messages.base import BaseMessage
from langchain_core.messages import filter_messages, ToolMessage, AIMessage

from openAI_api import llm
from criteria_of_answers import system_informations_of_functions
from openAI_api import chat_completion_request
from callable_tools.identifying_type import tools_of_type
from callable_tools.helping_travel import tools_of_travel

from typing import (
    Optional, 
    TypedDict,
    Annotated, 
    Union, 
    Sequence,
    List,
    Any
) 
    
# 노드에 전달되는 state
class AgentState(TypedDict):
    messages : Annotated[list, add_messages] # 대화 history
    previous_result : Optional[str]                    # 이전 단계에서의 결과값 
    final_response : Optional[dict]                    # 사용자에게 전달되는 최종 메시지 
    tbti_of_user : Optional[str]             # 사용자 TBTI  
    filtering : Optional[dict]               # DB 검색 필터
    name_of_tools: list
    
def escape_json_strings(response):
    try:
        # JSON 문자열을 딕셔너리 자료형으로 변환
        response_dict = json.loads(response)
        return response_dict
    except Exception as e:
        print(f"Error escaping JSON strings: {e}")
        return response
    
def create_my_agent(
    model: LanguageModelLike,
    tools: Union[ToolExecutor, Sequence[BaseTool], ToolNode],
    checkpointer: Optional[BaseCheckpointSaver] = None
) -> CompiledGraph:
    
    # LLM 시스템 프롬프트 리스트 로드 - 작동되는 함수에 따라 시스템 프롬프트 내용 다름
    configuration_for_answers = system_informations_of_functions
    # 사용할 수 있는 모델 로드
    model_with_tools = model.bind_tools(tools)

    # 첫 노드가 필요해서 만듦
    def save_user_info(state: AgentState):
        return {'tbti_of_user': state["tbti_of_user"]}
    
    # 검색 필터를 생성해야 하는지 파악
    def should_create_filter(state: AgentState):
        filtering = state['filtering']
        tbti = state['tbti_of_user']

        if filtering != None and tbti != None:
            return "start talking"
        elif filtering == None and tbti != None:
            return "create filter"
        else:
            return "start talking"
    
    # 새로운 검색 필터 생성 노드
    def generate_new_filter(state: AgentState):
        filtering = {}
        available_tools = []

        # 사용자 여행 유형 가져오기
        tbti = state['tbti_of_user']
        try:
            if tbti not in ['AIEU', 'AIEP', 'AIFU', 'AIFP', 'ASEU', 'ASEP', 'ASFU', 'ASFP', 'CIEU', 'CIEP', 'CIFU', 'CIFP', 'CSEU', 'CSEP', 'CSFU', 'CSFP']:
                raise Exception("전달받은 TBTI 유형이 존재하지 않습니다.")

            # 여행 유형에 따른 검색 필터 및 호출시킬 도구 준비
            tbti = list(tbti)
            for one_type in tbti:
                match one_type:
                    case 'A':
                        filtering["mood"] = "(mood == 0)"
                    case 'C':
                        filtering["mood"] = "(mood == 1)"
                    case 'I':
                        available_tools.append("check_companion_animal")
                    case 'P':
                        pass
                    case 'S':
                        available_tools.append("check_child")
                        available_tools.append("check_companion_animal")
                    case 'E':
                        available_tools.append("check_distance")
                    case 'F':
                        filtering["parking"] = "(parking == true)"
                    case 'U':
                        filtering["reservation"] = "(reservation == true)"

        except Exception as e:
            print(e, " 올바른 TBTI 유형을 전달하세요.")

        return {"filtering": filtering, "name_of_tools": available_tools}
    

    # 여행 유형에 맞는 추가 질문이 가능한 모델 만들기
    def make_model_with_tools(state: AgentState):
        name_of_tools = state["name_of_tools"]
        system_message = 'Ask a question in Korean one by one.'

        # 호출할 추가 도구 리스트 및 질문 가져오기
        for name in name_of_tools:
            added_msg = tools_of_type[name]["added_system_message"] + ' '
            system_message = system_message + added_msg

        messages = [
            ("system", f"{system_message}")
        ]

        return {"messages": messages}
    
    # 사용할 AI 모델 로드 및 AI 답변 처리
    def talk_to_model(state: AgentState):
        response = model_with_tools.invoke(state['messages'])
        last_response = response.content.strip("<>() ").replace('\"', '\'')
        last_response = f'{{\"answer\": \"{last_response}\", \"place\": null}}'

        # AI 답변을 json 형식의 문자열로 만들어 previous_result에 저장 / 답변 history에 저장 
        return {"previous_result" : last_response , "messages" : [response]}

    # 검색 필터를 추가할 지, 검색 결과를 통한 답변을 생성할 지 파악
    def should_make_answer(state: AgentState):
        name_of_tools = tools_of_type.keys()
        tool_messages = state["messages"][-1]

        if tool_messages.name in name_of_tools:
            return "add type"
        else:
            return "make answer"
    
    # 여행 취향을 파악하기 위한 추가 질문 답변 처리
    def process_type_result(state: AgentState):
        filtering = state["filtering"]
        messages = state["messages"]
        
        ai_message = filter_messages(messages, include_types=[AIMessage])[-1]
        tool_call_ids = [item['id'] for item in ai_message.tool_calls]

        # tool_message 가져오기
        tool_messages =  filter_messages(messages, include_types=[ToolMessage], include_ids=tool_call_ids)
        for tool in tool_messages:
            content = tool.content
            if content != 'null':
                split_result = content.split(',')
                print(split_result)
                filtering[split_result[0]] = split_result[1]

        print(filtering)
        return {"filtering": filtering}

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
        response_format = configuration_for_answers[name_of_functions_called]['response_format']

        # 결과 참고해서 LLM 답변 생성
        messages = [
            {"role":"system", "content": f"{system_prompt}"},
            {"role": "user", "content":f"{reference}"}
        ]
        
        llm_response = chat_completion_request(
            messages=messages,
            response_format=response_format
        ).choices[0].message.content   

        return {"previous_result": llm_response}
    
    # 도구를 작동 시킬 지 파악
    def should_continue(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        
        # 함수 호출이 없으면 바로 사용자에게 리턴
        if not last_message.tool_calls:
            return "pass"
        # 있으면 워크플로우 지속
        else:
            return "work"

    def post_processing_of_answer(state: AgentState):
        ai_answer = state["previous_result"]
        escaped_response = escape_json_strings(ai_answer)
        return {"final_response" : escaped_response}

        
    # 새로운 그래프 정의
    workflow = StateGraph(AgentState)

    # 각 노드 생성
    workflow.add_node("start-node", save_user_info)
    workflow.add_node("generate-filter", generate_new_filter)
    workflow.add_node("make-model", make_model_with_tools)
    workflow.add_node("talk-to-human", talk_to_model)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("add-filter", process_type_result)
    workflow.add_node("respond", respond_after_calling_tools)
    workflow.add_node("json-processing", post_processing_of_answer)

    # 그래프 진입 포인트 설정
    workflow.set_entry_point("start-node")

    workflow.add_conditional_edges(
        "start-node",
        should_create_filter,
        {
            "start talking": "talk-to-human",
            "create filter": "generate-filter",
        },
    )
    
    workflow.add_edge("generate-filter", "make-model")
    workflow.add_edge("make-model", "talk-to-human")

    workflow.add_conditional_edges(
        "talk-to-human",
        should_continue,
        { 
            "work": "tools",
            "pass": "json-processing"
        }
    )

    workflow.add_conditional_edges(
        "tools",
        should_make_answer,
        {
            "add type": "add-filter",
            "make answer": "respond"
        }
    )

    workflow.add_edge("add-filter", "talk-to-human")
    workflow.add_edge("respond", "json-processing")
    workflow.add_edge("json-processing", END)

    graph = workflow.compile(
        checkpointer=checkpointer
    )

    return graph