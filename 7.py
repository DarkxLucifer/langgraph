from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
load_dotenv()

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

llm = ChatGroq(model="llama-3.1-8b-instant")

def chat_node(state: ChatState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)
chatbot = graph.compile()

state: ChatState = {"messages": []}

print("Chatbot started (type 'exit' to stop, 'history' to view chat)\n")

while True:
    user_input = input("User: ")

    if user_input.lower() == "exit":
        break

    if user_input.lower() == "history":
        print("\n--- Chat History ---")
        for msg in state["messages"]:
            role = "User" if isinstance(msg, HumanMessage) else "AI"
            print(f"{role}: {msg.content}")
        print("--------------------\n")
        continue

    # ✅ Append new message to existing state before invoking
    state["messages"].append(HumanMessage(content=user_input))
    state = chatbot.invoke(state)

    print("AI:", state["messages"][-1].content)