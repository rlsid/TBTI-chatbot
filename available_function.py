from access_milvusDB import db
from openAI_api import embedding
from openAI_api import chat_completion_request

## 여행지 추천 함수
def recommand_travel_destination(question, location):
    
    milvus = db

    # 사용자 질문 벡터화
    vector  =  embedding(question)

    # 필터링 생성 후 테이블 검색 진행
    area_name = milvus.make_filtering(location)
    filtering = f"area_name == '{area_name}'" 
    results_localCreator, results_nowLocal = milvus.search_all_tables(embedding=vector, filtering=filtering)

    # 쿼리 결과 합치기
    total_results = milvus.get_formatted_results(results_localCreator, results_nowLocal)

    # 결과 참고해서 LLM 답변 생성
    messages = [
        {"role": "user", "content":f"사용자 질문: {question} \n reference: {total_results}" }
    ]

    system_prompt = """
    - 다음과 같은 형식의 JSON 객체로 추천할 만한 여행지 목록을 생성해 주세요. 
    - reference의 정보를 사용하며, reference 이외의 정보는 사용하지 않습니다. 당신이 알고있는 정보는 사용하지 않습니다.
    - 각 장소는 이름, 위치, 카테고리, 설명, 그리고 리다이렉션 URL을 포함합니다. 
    - JSON 객체는 다음과 같은 구조를 가져야 합니다:
    {"answer": "장소를 추천한다는 짧은 추천의 말이 들어갑니다","place": [{"place_name": "장소 이름","location": "장소의 위치","category": "장소의 카테고리","description": "장소에 대한 간단한 설명","redirection_url": "장소에 대한 추가 정보를 제공하는 URL"},...]}
    - 이 형식에 맞게 명소 최대 5곳을 추천해 주세요.
    """
    messages.append({"role":"system", "content": f"{system_prompt}"})

    llm_response = chat_completion_request(
        messages=messages,
        response_format={"type":"json_object"}
    )
    
    return llm_response.choices[0].message.content


## 여행 계획 생성 함수
def create_travel_plan(question, location, duration):
    
    milvus = db
    
    vector = embedding(question)
    
    filtering = milvus.make_filtering(location)
    results_localCreator, results_nowLocal = milvus.search_all_tables(embedding=vector, filtering=filtering)

    total_results = milvus.get_formatted_results(results_localCreator, results_nowLocal)
    
    messages = [
        {"role": "system", "content": "당신은 여행을 계획하는 데에 도움을 주는 chatbot 'TBTI'입니다.\n당신의 역할은 사용자의 질문에서 사용자가 여행하려는 여행 지역과 여행 기간을 파악하고 reference를 참고하여 여행 계획을 생성하는 것입니다.\nreference 참고 자료에서 쓸만한 정보가 부족할 때, 당신이 기존에 알고 있는 정보를 이용하여 답변하세요."},
        {"role": "user", "content": f"사용자 질문: {question} \n 여행 지역: {location} \n 여행 기간: {duration} \n reference: {total_results}"}
    ]
    
    # 프롬프트 추가(그냥 잘 보이게 추가한 것)
    system_prompt = """
    여행 계획을 세울 때에는 다음의 조건을 꼭 만족해야 합니다.
    1. 각 날짜별로 방문할 장소를 추천합니다. 이때 날짜별로 추천하는 장소의 개수는 3개여야 합니다.
    2. 같은 날짜에 방문하는 장소끼리의 거리는 10KM 이내여야 합니다.
    3. 모든 날짜를 통틀어 추천되는 장소는 겹치면 안 됩니다.
    4. 장소의 카테고리가 '카페/디저트'인 장소는 한 날짜에 하나보다 많으면 안 됩니다.
    5. 각 장소는 위치, 카테고리, 설명, 운영정보가 있어야 합니다.
    6. 설명은 장소의 키워드를 이용하여 만들어 주세요.
    7. 운영정보 값이 reference에 존재할 때만 운영 정보를 제공해 주세요. 운영정보값은 reference 이외의 값에서 가져오지 않습니다.
    """
    messages.append({"role": "system", "content": system_prompt})
    
    llm_response = chat_completion_request(messages).choices[0].message.content
    
    return llm_response


def reserve_place(question, location=None, place_name=None):
    
    # DB 연결 및 질문 임베딩
    milvus = db
    vector = embedding(question)

    #filtering 생성
    filters = ["(reservation == true)"]
    if location:
        area_name = milvus.make_filtering(location)
        filters.append(f"(area_name == '{area_name}')")
    elif place_name:
        place_name = place_name.replace(' ', '%')
        filters.append(f"(place_name like '{place_name}%')")
    filtering = ' and '.join(filters)

    # 쿼리 진행
    results_localCreator, results_nowLocal = milvus.search_all_tables(embedding=vector, filtering=filtering)
    
    # 쿼리 결과 합치기
    total_results = milvus.get_formatted_results(results_localCreator, results_nowLocal)

    # 결과 참고해서 LLM 답변 생성
    messages = [
        {"role": "user", "content":f"사용자 질문: {question} \n reference: {total_results}" }
    ]

    system_prompt = """
    - 다음과 같은 형식의 JSON 객체로 예약 가능한 여행지 목록을 생성해 주세요. 
    - reference의 정보를 사용하며, reference 이외의 정보는 사용하지 않습니다.
    - 각 장소는 이름, 위치, 카테고리, 설명, 그리고 리다이렉션 URL을 포함합니다. 
    - JSON 객체는 다음과 같은 구조를 가져야 합니다:
    {"answer": "예약 가능한 장소를 알려준다는 짧은 말이 들어갑니다","place": [{"place_name": "장소 이름","location": "장소의 위치","category": "장소의 카테고리","description": "장소에 대한 간단한 설명","redirection_url": "장소에 대한 추가 정보를 제공하는 URL"},...]}
    - 사용자가 예약 가능한 여러 장소를 알고 싶어할 경우 최대 5곳을 알려주고, 특정 하나의 장소일 경우 1곳만 알려줍니다.
    """
    messages.append({"role": "system", "content": system_prompt})

    llm_response = chat_completion_request(
        messages=messages,
        response_format={"type":"json_object"}
    )
    
    return llm_response.choices[0].message.content
    

print("available_function 모듈 로드")
        