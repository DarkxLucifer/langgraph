from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class Node(TypedDict):
    Height: int
    Weight: int
    BMI: float
    LABEL: str


def Bmi(state: Node) -> Node:
    height = state["Height"]
    weight = state["Weight"]
    state["BMI"] = height / weight
    return state


def label(state: Node) -> Node:
    BMI = state["BMI"]

    if BMI > 50:
        print("BMI greater than 50")
    else:
        print("BMI less than 50")

    return state


graph = StateGraph(Node)
graph.add_node("BMI", Bmi)
graph.add_node("label", label)

graph.add_edge(START, "BMI")
graph.add_edge("BMI", "label")
graph.add_edge("label", END)

workflow = graph.compile()


def printy(workflow):
    png_data = workflow.get_graph().draw_mermaid_png()

    with open("graph.png", "wb") as f:
        f.write(png_data)

    print("Graph saved as graph.png")


# 👇 only runs if s1.py is executed directly
if __name__ == "__main__":
    printy(workflow)
    yo = workflow.invoke({"Weight": 50, "Height": 5})
    print(yo)