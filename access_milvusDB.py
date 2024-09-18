from pymilvus import MilvusClient

URI = 'http://awsip:19530'

class MilvusDB:
    def __init__(self, uri=URI):
        self.uri = uri
        self.client = self.connect()

    # DB 연결 함수
    def connect(self):
        try:
            client = MilvusClient(self.uri)
            print('Milvus 연결 성공')
        except Exception as e:
            print('Milvus 연결 실패: ', e)
            client = None
        return client   

    # DB 재연결 함수 (연결이 끊어진 경우 재연결 시도)
    def reconnect(self):
        if not self.client:
            print("재연결 시도 중...")
            self.client = self.connect()
        return self.client  
    '''
    # 행정 구역을 기준으로 검색 지역 설정
    def select_searching_partition(self, input):
        from openai import OpenAI

        messages = [
            {"role": "system", "content":"사용자의 질문 속의 장소가 아래의 한국 행정 구역 중 어디에 속하는지 다음과 같이 단어로만 대답해주세요.\n ex. 강원특별자치도 \n\n - 한국 행정 구역 : \n서울특별시, 부산광역시, 인천광역시, 대구광역시, 대전광역시, 광주광역시, 울산광역시, 세종특별자치시, 경기도, 충청북도, 충청남도, 전라남도, 경상북도, 경상남도, 강원특별자치도, 전북특별자치도, 제주특별자치도"},
            {"role": "user", "content": f"{input}"}
        ]
        response = chat_completion_request(
            messages=messages
        )
        return response.choices[0].message.content
    '''

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
    def search_all_tables(self, embedding, filtering, top_k=10):
        results_localCreator = self.search_table('kstartup_travel_sites', embedding, filtering, top_k)
        results_nowLocal = self.search_table('nowlocal_travel_sites', embedding, filtering, top_k)

        return results_localCreator, results_nowLocal

    # 쿼리 검색 결과 하나로 묶는 함수
    def get_formatted_results(self, results_localCreator, results_nowLocal):
        list_of_results = [results_localCreator[0], results_nowLocal[0]]
        #formatted_results = []
        formatted_results = ""

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
                text = f"- {result[num]['entity']['text']}"
                text += f"상세페이지: 'http://localhost:3000/detail/{result[num]['entity']['id']}'"
                formatted_results += text + "\n"
            
        return formatted_results
    
    # Milvus 연결 해제 함수
    def unconnect(self):
        if self.client:
            self.client.close()
            print("Milvus 연결 종료")
            self.client = None

# db 연결
db = MilvusDB()

print("accees_milvusDB 모듈 로드")
