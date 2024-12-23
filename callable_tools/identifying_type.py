from langchain_core.tools import tool

# 자녀가 있는지 없는지에 대한 답변에 작동되어 자녀 여부를 저장
@tool
def check_child(response: bool) -> str:
    """check whether the users are going to travel with their children or not and save their answers to the parameters.

      Args:
        response: if the user is going to travel whith their children, put 'true'. if not, put 'false'
    """
    if response == True:
      filter_string = 'child,(child == true)'
    else:
      filter_string = 'child,null'
    return filter_string


# 반려동물과 함께 여행하는지 아닌지에 대한 답변에 작동되어 답변 저장
@tool
def check_companion_animal(response: bool) -> str:
  """check whether the users are going to travel with their pets or not and save their answers to the parameters.

    Args:
      response: if the user is going to travel with a pet, put 'true'. if not, put 'false'.
  """

  if response == True:
    filter_string = 'animal,(animal == true)'
  else:
    filter_string = 'animal,null'
  return filter_string


# 장소가 근처 정류장과 얼마나 가까우면 좋을지에 대한 답변 저장
@tool
def check_distance(response: str) -> str:
  """Check the answer to how close the travel site is to the nearby stop. save their answers to the parameters.

    Args:
    response: You can save only one of the three answers. ex. '5분 이내' or '10분 이내' or '15분 이상'
  """
  if response in ['5분 이내', '10분 이내', '15분 이상']:
    filter_string = f"walking_distance,(walking_distance == '{response}')"
  else:
    filter_string = 'walking_distance,null'

  return filter_string


# 호출 가능한 함수 정보 저장 -> 사용자에게 할 추가 질문 생성에 쓰임
tools_of_type = {
    "check_child": {
        "func" : check_child,
        "added_system_message" : "Ask first if users travel with their children."
    },
    "check_companion_animal" : {
        "func" : check_companion_animal,
        "added_system_message" : "Ask first if users are going to travel with their pets."
    },
    "check_distance": {
        "func" : check_distance,
        "added_system_message" : "Ask first how close the place you want to visit should be to the bus stop."
    },
    "list_of_func" : [check_child, check_companion_animal, check_distance]
}