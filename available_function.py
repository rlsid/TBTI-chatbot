from access_miluvsDB import db
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
        {"role": "system", "content": "- 당신은 여행을 계획하는데 도움을 주는 chatbot 'TBTI'입니다. \n- 당신의 역할은 사용자의 질문에 reference를 바탕으로 답변하는 것 입니다.\n- reference 참고 자료에서 쓸만한 정보가 부족할 때, 당신이 기존에 알고 있는 정보를 이용하여 답변하세요.\n- 장소에 대한 정보를 전달할 때는 아래의 조건을 만족해야 합니다.\n\n조건:\n1. 전달 장소 개수: 5개 \n2. 각 장소는 위치, 카테고리, 설명, 운영정보가 있어야 합니다.\n3. 설명은 장소의 키워드를 이용하여 만들어주세요. \n4.운영 정보가 존재할 때만, 운영 정보를 제공해주세요."},
        {"role": "user", "content":f"사용자 질문: {question} \n reference: {total_results}" }
    ]
    llm_response = chat_completion_request(messages=messages).choices[0].message.content
    
    return llm_response


## 여행 계획 생성 함수
