from pymilvus import MilvusClient

URI = 'http://43.201.50.124:19530'

class MilvusDB:
    def __init__(self, uri=URI):
        self.uri = uri
        self.client = self.connect()

    # DB 연결 함수
    def connect(self):
        try:
            client = MilvusClient(self.uri)

        except Exception as e:
            print('Milvus 연결 실패: ', e)

        #print('Milvus 연결 성공')
        return client   

    # 필터링 생성 함수(= 현재 행정 구역 기준으로 필터링)
    def make_filtering(self, input):
        from openAI_api import chat_completion_request

        messages = [
            {"role": "system", "content":"사용자의 질문 속의 장소가 아래의 한국 행정 구역 중 어디에 속하는지 다음과 같이 단어로만 대답해주세요.\n ex. 강원특별자치도 \n\n - 한국 행정 구역 : \n서울특별시, 부산광역시, 인천광역시, 대구광역시, 대전광역시, 광주광역시, 울산광역시, 세종특별자치시, 경기도, 충청북도, 충청남도, 전라남도, 경상북도, 경상남도, 강원특별자치도, 전북특별자치도, 제주특별자치도"},
            {"role": "user", "content": f"{input}"}
        ]
        response = chat_completion_request(
            messages=messages
        )
        return response.choices[0].message.content

    # 단일 테이블 검색 함수
    def search_table(self, table_name, embedding, filtering, top_k):
        search_params = {"metric_type": "IP", "params": {}}
        results = self.client.search(
            collection_name=table_name,
            data=[embedding],
            filter=f"{filtering}",
            anns_field="embedding",
            search_params=search_params,
            output_fields=["id", "text", "reservation", "place_name", "category"],
            limit=top_k
        )
        return results
    
    # 여러 테이블 검색 함수
    def search_all_tables(self, embedding, filtering):
        results_localCreator = self.search_table('kstartup_travel_sites', embedding, filtering, top_k=10)
        results_nowLocal = self.search_table('nowlocal_travel_sites', embedding, filtering, top_k=10)

        return results_localCreator, results_nowLocal

    # 쿼리 검색 결과 하나로 묶는 함수
    def get_formatted_results(self, results_localCreator, results_nowLocal):
        list_of_results = [results_localCreator[0], results_nowLocal[0]]
        formatted_results = []
        #formatted_results = ""

        for result in list_of_results:
            '''
            length = len(result)
            for num in range(length):
                data = {
                    'place_name': result[num]['entity']['place_name'],
                    'category': result[num]['entity']['category'],
                    'redirection_url': f"http://localhost:3000/detail/{result[num]['entity']['id']}"
                }
                formatted_results.append(data)
            ''' 
            
            length = len(result)
            for num in range(length):
                text = result[num]['entity']['text']
                text += f"상세페이지: 'http://localhost:3000/detail/{result[num]['entity']['id']}'"
                formatted_results += text + "\n\n"
            
        return formatted_results
    
    # milvus 연결 끊기
    def unconnect(self):
        self.client.close()

# db 연결
db = MilvusDB()

print("accees_milvusDB 모듈 로드")