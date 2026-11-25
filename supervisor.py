from typing import Literal
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START, END
from workers import policy_agent, room_agent, general_agent

llm = init_chat_model("openai:gpt-4o-mini")
WORKERS = {"policy": policy_agent, "room": room_agent, "general": general_agent}

def supervisor(state: MessagesState) -> Literal["policy", "room", "general"]:
    q = state["messages"][-1].content
    pick = llm.invoke(f"다음 질문을 policy / room / general 중 하나로만 분류: {q}").content.strip().lower()
    return pick if pick in WORKERS else "general"

def make_node(agent):
    def node(state: MessagesState):
        r = agent.invoke({"messages": state["messages"]}, {"recursion_limit": 8})
        return {"messages": [r["messages"][-1]]}
    return node

g = StateGraph(MessagesState)
for k, a in WORKERS.items():
    g.add_node(k, make_node(a)); g.add_edge(k, END)
g.add_conditional_edges(START, supervisor)
graph = g.compile()

if __name__ == "__main__":
    for q in ["재택 몇 번 돼?", "9명 회의실", "파이썬 리스트 정렬법"]:
        print(q, "->", graph.invoke({"messages": [("user", q)]})["messages"][-1].content)
