from langchain_core.tools import tool
from access_milvusDB import db
from openAI_api import embed

# 여행지 추천 함수
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

    milvus = db

    # 사용자 질문 벡터화
    vector  =  embed.embed_query(question)

    # 검색 필터링 생성 후 테이블 검색 진행
    filtering = f"area_name == '{area}'"
    results_localCreator, results_nowLocal = milvus.search_all_tables(embedding=vector, filtering=filtering)

    # 검색 결과 합치기
    total_results = milvus.get_formatted_results(results_localCreator, results_nowLocal)

    final_data= f"""
        - 당신은 사용자가 원하는 여러 장소의 정보를 알려줍니다.
        - reference의 정보를 사용하며, reference 이외의 당신이 알고있는 정보는 사용하지 않습니다.
        - 사용자에게 전달하는 장소는 반드시 장소 이름, 위치, 장소 카테고리, 설명(장소 키워드와 해시태그를 이용해서 생성), 상세페이지 URL만을 존재합니다.
        - 5개의 장소만 알려줍니다.
        - reference : {total_results}
    """

    return final_data


# 여행 계획 생성 함수
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

    milvus = db

    # 사용자 질문 벡터화
    vector  =  embed.embed_query(question)

    # 검색 필터링 생성 후 테이블 검색 진행
    filtering = f"area_name == '{area}'"
    results_localCreator, results_nowLocal = milvus.search_all_tables(embedding=vector, filtering=filtering)

    # 검색 결과 합치기
    total_results = milvus.get_formatted_results(results_localCreator, results_nowLocal)

    final_data = f"""
        - 당신은 여행을 계획하는데 도움을 줍니다. 사용자가 여행하려는 지역과 여행 기간을 파악하고 여행 계획을 생성하세요.
        - reference의 정보를 사용하며, reference 이외의 당신이 알고있는 정보는 사용하지 않습니다.
        - 여행 계획을 세울 때에는 다음의 조건을 꼭 만족해야 합니다.
            1. 각 날짜별로 방문할 장소를 추천합니다. 이때 날짜별로 추천하는 장소의 개수는 3개여야 합니다.
            2. 같은 날짜에 방문하는 장소끼리의 거리는 10KM 이내여야 합니다.
            3. 모든 날짜를 통틀어 추천되는 장소는 겹치면 안 됩니다.
            4. 장소의 카테고리가 '카페/디저트'인 장소는 한 날짜에 하나보다 많으면 안 됩니다.
            5. 각 장소는 이름, 위치, 카테고리, 설명이 있어야 합니다.
            6. 설명은 장소의 키워드를 이용하여 만들어 주세요.
        - reference : {total_results} 
    """
    return final_data

callable_tools = [recommand_travel_destination, create_travel_plan]