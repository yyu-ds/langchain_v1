"""
Demo 06: Hello LangGraph
Basic example showing how to build a simple graph with LangGraph.
"""

from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()


# Define the state
class State(TypedDict):
    messages: list
    response: str


def chat_node(state: State) -> State:
    """Process messages and generate response."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    response = llm.invoke(state["messages"])
    return {"response": response.content}


def main():
    # Build the graph
    workflow = StateGraph(State)

    # Add nodes
    workflow.add_node("chat", chat_node)

    # Add edges
    workflow.add_edge(START, "chat")
    workflow.add_edge("chat", END)

    # Compile the graph
    app = workflow.compile()

    # Run the graph
    result = app.invoke({
        "messages": [HumanMessage(content="What is LangGraph in one sentence?")],
        "response": "",
    })

    print(f"Response: {result['response']}")


if __name__ == "__main__":
    main()
