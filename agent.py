"""
ReAct Agent Core Module
Reasoning + Acting with LangGraph, OpenAI, and Tavily
"""

import os
import json
import warnings
from typing import Annotated, Sequence, TypedDict, Generator

warnings.filterwarnings("ignore")

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
)
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """Accumulated conversation + tool context for a single run."""
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

def make_search_tool(tavily_api_key: str):
    os.environ["TAVILY_API_KEY"] = tavily_api_key
    search = TavilySearchResults()

    @tool
    def search_tool(query: str):
        """
        Search the web for information using Tavily API.

        :param query: The search query string
        :return: Search results related to the query
        """
        return search.invoke(query)

    return search_tool


@tool
def recommend_clothing(weather: str) -> str:
    """
    Returns a clothing recommendation based on the provided weather description.

    :param weather: A brief description of the weather (e.g., "Overcast, 64.9°F")
    :return: A string with clothing recommendations suitable for the weather
    """
    w = weather.lower()
    if "snow" in w or "freezing" in w:
        return "Wear a heavy coat, gloves, and boots."
    elif "rain" in w or "wet" in w:
        return "Bring a raincoat and waterproof shoes."
    elif "hot" in w or "85" in w:
        return "T-shirt, shorts, and sunscreen recommended."
    elif "cold" in w or "50" in w:
        return "Wear a warm jacket or sweater."
    else:
        return "A light jacket should be fine."


# ---------------------------------------------------------------------------
# Agent builder
# ---------------------------------------------------------------------------

def build_agent(openai_api_key: str, tavily_api_key: str):
    """Compile and return a LangGraph ReAct agent."""
    os.environ["OPENAI_API_KEY"] = openai_api_key

    search_tool = make_search_tool(tavily_api_key)
    tools = [search_tool, recommend_clothing]
    tools_by_name = {t.name: t for t in tools}

    model = ChatOpenAI(model="gpt-4o-mini")

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant that thinks step-by-step and uses tools when needed.

When responding to queries:
1. First, think about what information you need
2. Use available tools if you need current data or specific capabilities
3. Provide clear, helpful responses based on your reasoning and any tool results

Always explain your thinking process to help users understand your approach."""),
        MessagesPlaceholder(variable_name="scratch_pad"),
    ])

    model_react = chat_prompt | model.bind_tools(tools)

    # --- Nodes ---
    def call_model(state: AgentState):
        response = model_react.invoke({"scratch_pad": state["messages"]})
        return {"messages": [response]}

    def tool_node(state: AgentState):
        outputs = []
        for tool_call in state["messages"][-1].tool_calls:
            result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            outputs.append(
                ToolMessage(
                    content=json.dumps(result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

    def should_continue(state: AgentState):
        last = state["messages"][-1]
        return "continue" if last.tool_calls else "end"

    # --- Graph ---
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.add_edge("tools", "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"continue": "tools", "end": END},
    )
    workflow.set_entry_point("agent")
    graph = workflow.compile()

    return graph


# ---------------------------------------------------------------------------
# Streaming helper
# ---------------------------------------------------------------------------

def stream_agent(graph, query: str) -> Generator[dict, None, None]:
    """
    Stream agent steps for a given query.
    Yields dicts with keys: role, content, tool_calls
    """
    inputs = {"messages": [HumanMessage(content=query)]}
    for step in graph.stream(inputs, stream_mode="values"):
        msg = step["messages"][-1]
        if isinstance(msg, HumanMessage):
            yield {"role": "human", "content": msg.content, "tool_calls": []}
        elif isinstance(msg, AIMessage):
            calls = [
                {"name": tc["name"], "args": tc["args"]}
                for tc in (msg.tool_calls or [])
            ]
            yield {"role": "ai", "content": msg.content, "tool_calls": calls}
        elif isinstance(msg, ToolMessage):
            yield {"role": "tool", "content": msg.content, "name": msg.name, "tool_calls": []}
