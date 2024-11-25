import json
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from typing import Annotated
from access_milvusDB import database
from openAI_api import embedding

@tool
def recommand_travel_destination(question : str, location : str, area : str, filtering: Annotated[dict, InjectedState('filtering')]) -> str:
    """
    recommand the various places that user wants to know or to travel
    It only works when user wants to know the various places.
    It doesn't work when user told to plan the trip and when user told to reserve the place.

    Args:
        question: input the user's questions.
        location: input the area of Korea to travel, e.g. 서울 or 부산 or 대구 or 강원도
        area: Enter only the following words to indicate where the place in the user's question belongs to the following Korean administrative districts. e.g. 강원특별자치도 
              - 한국 행정 구역 : 서울특별시, 부산광역시, 인천광역시, 대구광역시, 대전광역시, 광주광역시, 울산광역시, 세종특별자치시, 경기도, 충청북도, 충청남도, 전라남도, 경상북도, 경상남도, 강원특별자치도, 전북특별자치도, 제주특별자치도
    """

    milvus = database

    # 사용자 질문 벡터화
    vector  =  embedding(question)

    # 필터링 생성 후 테이블 검색 진행
    milvus_filter = None
    area_filter = f"area_name == '{area}'"
    
    if bool(filtering):
        area_filter = area_filter + ' && '
        another = ' && '.join(filtering.values())
        milvus_filter = area_filter + another
    else:
        milvus_filter = area_filter
    
    results_localCreator, results_nowLocal = milvus.search_all_tables(embedding=vector, filtering=milvus_filter, top_k=5)

    # 쿼리 결과 합치기
    total_results = milvus.get_formatted_results(results_localCreator, results_nowLocal)

    return f"user question: {question} \n\nreference: \n{total_results}"


@tool
def create_travel_plan(question : str, location : str, area : str, duration : str, filtering: Annotated[dict, InjectedState('filtering')]) -> json:
    """
    works when the user wants to plan a trip

    Args:
        question: input the user's questions.
        location: input the area of Korea to travel, e.g. 서울 or 부산 or 대구 or 강원도
        area: Enter only the following words to indicate where the place in the user's question belongs to the following Korean administrative districts. e.g. 강원특별자치도 
              - 한국 행정 구역 : 서울특별시, 부산광역시, 인천광역시, 대구광역시, 대전광역시, 광주광역시, 울산광역시, 세종특별자치시, 경기도, 충청북도, 충청남도, 전라남도, 경상북도, 경상남도, 강원특별자치도, 전북특별자치도, 제주특별자치도
        duration: the duration of the travel plans, If the user does not specify the duration of the trip, you should ask specifically about the duration of the trip. e.g. 하루, 1박 2일, 2박 3일
    """

    milvus = database
    
    vector = embedding(question)
    
    # 필터링 생성 후 테이블 검색 진행
    milvus_filter = None
    area_filter = f"area_name == '{area}'"
    
    if bool(filtering):
        area_filter = area_filter + ' && '
        another = ' && '.join(filtering.values())
        milvus_filter = area_filter + another
    else:
        milvus_filter = area_filter
    
    results_localCreator, results_nowLocal = milvus.search_all_tables(embedding=vector, filtering=milvus_filter)

    total_results = milvus.get_formatted_results(results_localCreator, results_nowLocal)
    
    return f"user question: {question} \ntravel loaction: {location} travel duration: {duration} \nreference: \n{total_results}"


@tool
def search_specific_place(question : str, place_name : str = None) -> json:
    """
    give the information of the specific places mentioned by the user.
    It works when a user question contains a name of specific place.
    It doesn't work when you recommand a place.

    Args:
        question: input the user's question as it is
        place_name: The name of the particular place that user wants to know or to reserve
    """
    
    milvus = database
    vector = embedding(question)

    # filtering 생성
    filtering = ''
    if place_name:
        place_name = place_name.replace(' ', '')
        char = f"%{place_name[0:4]}%"
        filtering = f"place_name like '{char}'"

    # 쿼리 진행
    results_localCreator, results_nowLocal = milvus.search_all_tables(embedding=vector, filtering=filtering, top_k=3)
    total_results = milvus.get_formatted_results(results_localCreator, results_nowLocal)

    return f"user question: {question} \n\nreference: \n{total_results}"
    

tools_of_travel = [recommand_travel_destination, create_travel_plan, search_specific_place]