import json
from langchain_core.tools import tool
from access_milvusDB import database
from openAI_api import chat_completion_request
from openAI_api import embedding

@tool
def recommand_travel_destination(question : str, location : str, area : str) -> str:
    """
    gives information on various places that user wants to know or to travel
    It doesn't work when user told to plan the trip and when user told to reserve the place.

    Args:
        question: input the user's question as it is
        location: The area of Korea, e.g. 서울 or 부산 or 대구
        area: Enter only the following words to indicate where the place in the user's question belongs to the following Korean administrative districts. e.g. 강원특별자치도 
              - 한국 행정 구역 : 서울특별시, 부산광역시, 인천광역시, 대구광역시, 대전광역시, 광주광역시, 울산광역시, 세종특별자치시, 경기도, 충청북도, 충청남도, 전라남도, 경상북도, 경상남도, 강원특별자치도, 전북특별자치도, 제주특별자치도
    """

    milvus = database

    # 사용자 질문 벡터화
    vector  =  embedding(question)

    # 필터링 생성 후 테이블 검색 진행
    filtering = f"area_name == '{area}'" 
    results_localCreator, results_nowLocal = milvus.search_all_tables(embedding=vector, filtering=filtering)

    # 쿼리 결과 합치기
    total_results = milvus.get_formatted_results(results_localCreator, results_nowLocal)

    # 결과 참고해서 LLM 답변 생성
    messages = [
        {"role": "user", "content":f"user's question: {question} \n reference: {total_results}" }
    ]

    system_prompt = """
    - Use the information from the reference and do not use any information other than the reference. Do not use the information you know.
    - Please create a list of recommended destinations with JSON objects in the following format.
    - JSON objects must have the following structure:
    {"answer": "put a short sentence that recommand the places", "place": [{"place_name": "The name of the place", "location": "A detailed location of a place", "category": "category of place", "description": "A brief description of the place. make by using the keywords of the place", "redirection_url": "A URL for more information about the place"},...]}
    - Please recommend up to five attractions according to this format.
    - The answer is in Korean
    """
    messages.append({"role":"system", "content": f"{system_prompt}"})

    llm_response = chat_completion_request(
        messages=messages,
        response_format={"type":"json_object"}
    ).choices[0].message.content
    
    return llm_response



@tool
def create_travel_plan(question : str, location : str, area : str, duration : str) -> str:
    """
    works when the user wants to plan a trip

    Args:
        question: input the user's question as it is
        location: The city of Korean, e.g. 서울 or 부산 or 대구
        area: Enter only the following words to indicate where the place in the user's question belongs to the following Korean administrative districts. e.g. 강원특별자치도 
              - 한국 행정 구역 : 서울특별시, 부산광역시, 인천광역시, 대구광역시, 대전광역시, 광주광역시, 울산광역시, 세종특별자치시, 경기도, 충청북도, 충청남도, 전라남도, 경상북도, 경상남도, 강원특별자치도, 전북특별자치도, 제주특별자치도
        duration: the duration of the travel plans, If the user does not specify the duration of the trip, you should ask specifically about the duration of the trip. e.g. 하루, 1박 2일, 2박 3일
    """

    milvus = database
    
    vector = embedding(question)
    
    filtering = f"area_name == '{area}'"
    results_localCreator, results_nowLocal = milvus.search_all_tables(embedding=vector, filtering=filtering)

    total_results = milvus.get_formatted_results(results_localCreator, results_nowLocal)
    
    messages = [
        {"role": "system", "content": "Your role is to create a travel plan by referring to the reference. Use the information from the reference and do not use any information other than the reference. Do not use the information you know. The answer is in Korean"},
        {"role": "user", "content": f"사용자 질문: {question} \n 여행 지역: {location} \n 여행 기간: {duration} \n reference: {total_results}"}
    ]
    
    # 프롬프트 추가(그냥 잘 보이게 추가한 것)
    system_prompt = """
    When you plan your trip, you must meet the following conditions.
    1. We recommend places to visit for each date. At this time, the number of places to recommend for each date should be three.
    2. The distance between the places you visit on the same date must be within 10KM.
    3. The recommended places across all dates should not overlap.
    4. Place categories on one date must not overlap and must vary.
    5. Each place must have a place name, location, and description.
    6. Please make the explanation using the keyword of the place.
    """
    messages.append({"role": "system", "content": system_prompt})
    llm_response = chat_completion_request(messages).choices[0].message.content
    
    return {"answer": llm_response, "place" : None}


callable_tools = [recommand_travel_destination, create_travel_plan]