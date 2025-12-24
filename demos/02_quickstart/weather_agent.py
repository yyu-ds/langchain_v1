from dotenv import load_dotenv
load_dotenv()

from dataclasses import dataclass
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy

# -------- Context --------
@dataclass
class Context:
    user_id: str

# -------- Tools --------
@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Get the user's location based on their ID."""
    return "SF" if runtime.context.user_id == "1" else "NYC"

# -------- Output --------
@dataclass
class Response:
    answer: str
    weather: str

# -------- Model --------
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
)

# -------- Agent --------
agent = create_agent(
    model=model,
    system_prompt=(
        "You are a helpful assistant. "
        "When asked about the weather, use the tools to find the location and weather information. "
        "In your response: "
        "1. Populate 'weather' with the raw weather information retrieved from the tool. "
        "2. Populate 'answer' with a friendly natural language response for the user."
    ),
    tools=[get_weather, get_user_location],
    context_schema=Context,
    response_format=ToolStrategy(Response),
    checkpointer=InMemorySaver(),
)

# -------- Run --------
config = {"configurable": {"thread_id": "t1"}}

res = agent.invoke(
    {"messages": [{"role": "user", "content": "what's the weather outside?"}]},
    config=config,
    context=Context(user_id="1"),
)

print(res["structured_response"])
