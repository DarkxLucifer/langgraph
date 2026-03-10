import time

from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

os.makedirs("database", exist_ok=True)
load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant")

class JokeState(TypedDict):
    topic: str
    joke: str
    explanation: str


def generate_joke(state: JokeState):
    prompt = f"Generate a joke about {state['topic']}"
    response = llm.invoke(prompt).content

    time.sleep(5)
    print("generating joke")
    return {"joke": response}


def generate_explanation(state: JokeState):
    prompt = f"Explain the following joke:\n{state['joke']}"
    print("yo")
    time.sleep(1)
    response = llm.invoke(prompt).content
    return {"explanation": response}


graph = StateGraph(JokeState)

graph.add_node("generate_joke", generate_joke)
graph.add_node("generate_explanation", generate_explanation)

graph.add_edge(START, "generate_joke")
graph.add_edge("generate_joke", "generate_explanation")
graph.add_edge("generate_explanation", END)
conn = sqlite3.connect("database/jokes.db")
conn = sqlite3.connect(
    "database/jokes.db",
    check_same_thread=False
)
checkpointer = SqliteSaver(conn)

workflow=graph.compile(checkpointer=checkpointer)

config1 = {"configurable": {"thread_id": "1"}}
state = workflow.get_state(config1)

if state:
    print("Resuming from checkpoint")
    result = workflow.invoke(None, config=config1)
else:
    print("Starting new run")
    result = workflow.invoke({'topic': 'pizza'}, config=config1)

print(result)
state = workflow.get_state(config1)
print(state)

