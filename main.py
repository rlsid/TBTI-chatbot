import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from access_milvusDB import db
from openAI_api import chat_completion_request
from available_function import recommand_travel_destination
from available_function import create_travel_plan
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

class AiResponse(BaseModel):
    response: str


# 함수 정의 가져오는 함수
def get_defined_function():
    try:
        with open('tools.json', encoding='UTF8') as f:
            tools = json.load(f)
        return tools
    
    except Exception as e:
        print('json 파일 로드하기 실패:', e)
        raise HTTPException(status_code=500, detail="JSON 파일 로드 실패")

# 질문을 처리하는 API 엔드포인트
@app.post("/ask-ai/", response_model=AiResponse)
async def ask_ai(request: QuestionRequest):
    question = request.question  # JSON에서 question 필드 추출

    # 정의된 함수 정보 가져오기
    tools = get_defined_function()

    # 사용자 질문
    messages = [
        {"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."},
        {"role": "user", "content": f"{question}"}
    ]

    try:
        chat_response = chat_completion_request(
            messages, tools=tools
        )

        assistant_message = chat_response.choices[0].message
        messages.append(assistant_message)

        tool_calls = assistant_message.tool_calls

        if tool_calls:
            available_functions = {
                "recommand_travel_destination": recommand_travel_destination,
                "create_travel_plan": create_travel_plan
            }
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions.get(function_name)
                function_args = json.loads(tool_call.function.arguments)

                # function_args에 question이 없는 경우에만 question 추가
                if 'question' not in function_args:
                    function_response = function_to_call(
                        question=question,
                        **function_args
                    )
                else:
                    function_response = function_to_call(
                        **function_args
                    )

                return {"response": function_response}
        else:
            return {"response": assistant_message.content}
    except Exception as e:
        print("에러 발생: ", e)
        raise HTTPException(status_code=500, detail="AI 처리 중 오류 발생")
    finally:
        # DB 연결 끊기
        db.unconnect()


# 포트수정가능
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)


