from access_milvusDB import db
from openAI_api import embedding
from openAI_api import chat_completion_request

## 여행지 추천 함수
def recommand_travel_destination(question, location):
    
    milvus = db

    # 사용자 질문 벡터화
    vector  =  embedding(question)

    # 필터링 생성 후 테이블 검색 진행
    filtering = milvus.make_filtering(location)
    results_localCreator, results_nowLocal, results_nature = milvus.search_all_tables(embedding=vector, filtering=filtering)

    # 쿼리 결과 합치기
    total_results = milvus.get_formatted_results(results_localCreator, results_nowLocal, results_nature)

    
    # 결과 참고해서 LLM 답변 생성
    messages = [
        {"role": "system", "content": "- 당신은 여행을 계획하는데 도움을 주는 chatbot 'TBTI'입니다. \n- 당신의 역할은 사용자의 질문에 reference를 바탕으로 답변하는 것 입니다.\n- reference 참고 자료에서 쓸만한 정보가 부족할 때, 당신이 기존에 알고 있는 정보를 이용하여 답변하세요.\n- 장소에 대한 정보를 전달할 때는 아래의 조건을 만족해야 합니다.\n\n조건:\n1. 전달 장소 개수: 5개 \n2. 각 장소는 위치, 카테고리, 설명이 있어야 합니다.\n3. 설명은 장소의 키워드를 이용하여 만들어주세요."},
        {"role": "user", "content":f"사용자 질문: {question} \n reference: {total_results}" }
    ]
    llm_response = chat_completion_request(messages=messages).choices[0].message.content
    
    return llm_response


## 여행 계획 생성 함수
def create_travel_plan(question, location, duration):
    
    milvus = db
    
    vector = embedding(question)
    
    filtering = milvus.make_filtering(location)
    results_localCreator, results_nowLocal, results_nature = milvus.search_all_tables(embedding=vector, filtering=filtering)

    total_results = milvus.get_formatted_results(results_localCreator, results_nowLocal, results_nature)
    
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


print("available_function 모듈 로드")