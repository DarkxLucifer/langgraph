from langgraph.graph import StateGraph,START,END
from typing import TypedDict

class Node(TypedDict):
    Height: int
    Weight: int
    BMI: float
    LABEL: str
def Bmi(sate: Node) -> Node:
    Height=sate["Height"]
    Weight=sate["Weight"]
    sate["BMI"]=Height/Weight
    return sate
def label(sate: Node) -> Node:
    BMI=sate["BMI"]
    print(BMI)
    if BMI>50:
        print("BMI greater than 50")
    elif BMI<50:
        print("BMI less than 50")

graph = StateGraph(Node)
graph.add_node("BMI",Bmi)
graph.add_node("label",label)

graph.add_edge(START,"BMI")
graph.add_edge("BMI","label")
graph.add_edge("label",END)
workflow=graph.compile()
def printy(workflow):

    png_data = workflow.get_graph().draw_mermaid_png()

    with open("graph.png", "wb") as f:
        f.write(png_data)

print("Graph saved as graph.png")
yo=workflow.invoke({"Weight":50,"Height":5})
print(yo)
