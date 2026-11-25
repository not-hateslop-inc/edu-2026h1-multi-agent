from typing import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END

llm = init_chat_model("openai:gpt-4o-mini")

class State(TypedDict):
    draft: str
    feedback: str
    rounds: int
    next: str

def writer(s: State):
    d = llm.invoke(f"사내 공지 초안을 써라. 이전 피드백: {s.get('feedback', '없음')}\n주제: 6월 정기 교육 종료 안내").content
    return {"draft": d, "rounds": s.get("rounds", 0) + 1}

def reviewer(s: State):
    fb = llm.invoke(f"다음 공지를 검토해 문제가 없으면 OK, 있으면 고칠 점 한 줄:\n{s['draft']}").content
    done = fb.strip().startswith("OK") or s["rounds"] >= 3      # 최대 3라운드
    return {"feedback": fb, "next": "end" if done else "writer"}

g = StateGraph(State)
g.add_node("writer", writer); g.add_node("reviewer", reviewer)
g.add_edge(START, "writer"); g.add_edge("writer", "reviewer")
g.add_conditional_edges("reviewer", lambda s: s["next"], {"writer": "writer", "end": END})
out = g.compile().invoke({"rounds": 0, "feedback": "", "draft": "", "next": ""})
print(out["rounds"], "rounds\n", out["draft"])
