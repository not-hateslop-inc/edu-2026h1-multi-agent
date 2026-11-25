from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool

load_dotenv()

@tool
def search_policy(q: str) -> str:
    """사내 규정을 검색한다 (실습용 축약 데이터)."""
    data = {"재택": "재택 근무는 주 2회까지", "연차": "연차는 3일 전까지 신청", "식대": "야근 식대 1만 2천 원"}
    return next((v for k, v in data.items() if k in q), "해당 규정 없음")

@tool
def find_room(people: int) -> str:
    """인원에 맞는 회의실을 찾는다."""
    return "B 소회의실" if people <= 6 else "A 대회의실" if people <= 12 else "C 세미나실"

policy_agent = create_agent("openai:gpt-4o-mini", tools=[search_policy], system_prompt="사내 규정 담당. 도구 결과만 근거로 답한다.")
room_agent = create_agent("openai:gpt-4o-mini", tools=[find_room], system_prompt="회의실 담당.")
general_agent = create_agent("openai:gpt-4o-mini", tools=[], system_prompt="일반 질문 담당. 사내 정보는 모른다고 답한다.")
