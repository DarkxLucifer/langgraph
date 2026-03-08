from langgraph.graph import StateGraph,START,END
from typing import TypedDict
from s1 import printy

class Node(TypedDict):
    run:int
    four:int
    six:int
    ball:int
    bpb:float
    bpercentage:float
    srate:float
    summary: str


def bpb(state: Node) -> dict:
    value = state["ball"] / (state["four"] + state["six"])
    return {"bpb": value}


def bpercentage(state: Node) -> dict:
    value = ((state["four"] * 4 + state["six"] * 6) / state["run"]) * 100
    return {"bpercentage": value}


def srate(state: Node) -> dict:
    value = (state["run"] / state["ball"]) * 100
    return {"srate": value}


def summary(state: Node) -> dict:
    text = f"""
Strike Rate - {state['srate']}
Balls per boundary - {state['bpb']}
Boundary percent - {state['bpercentage']}
"""
    return {"summary": text}

graph = StateGraph(Node)
graph.add_node("Boundary_per_ball",bpb)
graph.add_node("Boundary_percentage",bpercentage)
graph.add_node("strick_rate",srate)
graph.add_node("summary",summary)

graph.add_edge(START,"Boundary_per_ball")
graph.add_edge(START,"Boundary_percentage")
graph.add_edge(START,"strick_rate")
graph.add_edge("Boundary_percentage","summary")
graph.add_edge("strick_rate","summary")
graph.add_edge("Boundary_per_ball","summary")
graph.add_edge("summary",END)

workflow=graph.compile()

printy(workflow)
intial_state = {
    'run': 100,
    'ball': 50,
    'four': 6,
    'six': 4
}

answer=workflow.invoke(intial_state)
print(answer["summary"])
