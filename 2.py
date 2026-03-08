from langgraph.graph import StateGraph,START,END
from typing import TypedDict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

class State(TypedDict):
    question:str
    Outline:str
    answer:str

model = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash'
)

def outline(sate: State)->State:
    question = sate['question']
    prompt = f'Write detailed Outline for following topic {question} fro blog'
    answer=model.invoke(prompt).content
    sate['Outline'] = answer
    return sate

def blog(sate: State)->State:
    question = sate['question']
    Outline = sate['Outline']
    prompt = f'Write detailed blog for following topic {question} with following outline\n{Outline}'
    answer=model.invoke(prompt).content
    sate['answer'] = answer
    return sate


graph = StateGraph(State)
graph.add_node("Outline",outline)
graph.add_node("blog",blog)
graph.add_edge(START,"Outline")

graph.add_edge("Outline","blog")
graph.add_edge("blog",END)


workflow=graph.compile()
png_data = workflow.get_graph().draw_mermaid_png()

with open("graph.png", "wb") as f:
    f.write(png_data)

inp={"question":"RDP Hijacking"}

answer=workflow.invoke(inp)
print("-------------Outline--------------")
print(answer["Outline"])
print("-------------Blog--------------")
print(answer['answer'])