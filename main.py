import os
import json

from access_miluvsDB import db
from openAI_api import chat_completion_request
from available_function import recommand_travel_destination

# 함수 정의 가져오는 함수
def get_defined_function():
    try:
        with open('tools.json', encoding='UTF8') as f:
            tools = json.load(f)
        return tools
    
    except Exception as e:
        print('json 파일 로드하기 실패:', e)


    
def main(question):

    # 정의된 함수 정보 가져오기
    tools = get_defined_function()

    # 사용자 질문
    messages = [
        {"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."},
        {"role": "user", "content": f"{question}"}
    ]

    chat_response = chat_completion_request(
        messages, tools=tools
    )

    assistant_message = chat_response.choices[0].message
    messages.append(assistant_message)
    print(assistant_message)

    tool_calls = assistant_message.tool_calls

    # 호출할 수 있는 함수가 존재한다면 함수 호출
    if tool_calls:
        available_functions = {
            "recommand_travel_destination": recommand_travel_destination
        }
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                question=question,
                location=function_args.get("location")
            )

            print(function_response)

            '''
            # 함수 리턴값이 LLM 최종 답변이라면 굳이 아래 형식으로 실행하지 않아도 될듯
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": function_response,
                }
            )

            final_response = openAI_api.chat.completions.create(
                model="gpt-4o",
                messages=messages,
            )
            print(final_response.choices[0].message.content)
            '''
    else:
        print(assistant_message)

question = "강릉 여행지 알려줘"
main(question)

#DB 연결 끊기
db.unconnect()






