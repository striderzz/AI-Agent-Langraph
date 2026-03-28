# ReAct Agent — LangGraph + GPT-4o-mini

A full-stack Python web application that brings the **ReAct (Reasoning + Acting)** framework to life. Built with LangGraph, LangChain, OpenAI, and Tavily, this project converts a Jupyter notebook into a production-ready app with a streaming Bootstrap UI and a standalone graph visualizer.

---

![UI Screenshot](screenshots/ui_dallas.png)

---

## What is ReAct?

**ReAct = Reasoning + Acting.** Instead of answering from training data alone, the agent:

1. **Thinks** — reasons about what it needs to know
2. **Acts** — calls external tools (web search, custom logic)
3. **Observes** — reads tool results and updates its reasoning
4. **Repeats** — until it has enough to give a complete answer

```
User Query
    │
    ▼
┌─────────────────┐   tool_calls?   ┌────────────────────┐
│   Agent Node    │ ── YES ───────► │    Tools Node      │
│  (GPT-4o-mini)  │ ◄── results ─── │  search + clothing │
└─────────────────┘                 └────────────────────┘
    │
    │  NO tool calls
    ▼
 Final Response → User
```

---

## Features

- **Live streaming UI** — each reasoning step (tool call, search result, final answer) streams to the browser in real time via Server-Sent Events (SSE)
- **Dark-mode Bootstrap 5 interface** — sidebar for API key config, color-coded message bubbles per role (You / Agent / Tool)
- **Two available tools** — Tavily web search for live data + a clothing recommendation engine
- **Graph visualizer** — separate `visualize_graph.py` script generates both an interactive HTML diagram (Pyvis) and a static PNG (Matplotlib)
- **Clean separation of concerns** — agent logic, web server, and UI are fully decoupled

---

## Project Structure

```
react_agent/
├── agent.py              # Core ReAct agent: LangGraph state machine + tool definitions
├── app.py                # Flask web server with SSE streaming endpoint
├── visualize_graph.py    # Standalone graph visualizer (PNG + interactive HTML)
├── requirements.txt      # Python dependencies
└── templates/
    └── index.html        # Bootstrap 5 dark-mode frontend
```

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/your-username/react-agent-langgraph.git
cd react-agent-langgraph
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the web app

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

### 4. Configure API keys in the sidebar

| Key | Where to get it | Cost |
|-----|----------------|------|
| OpenAI API Key | https://platform.openai.com/api-keys | Pay-per-use |
| Tavily API Key | https://app.tavily.com/sign-in | Free credits included |

Enter both keys in the left sidebar and click **Connect Agent**. You'll see a green **Agent Ready** badge.

### 5. Ask a question

Try:
- *"What's the weather in Dallas and what should I wear?"*
- *"Current weather in Mumbai, what to wear?"*
- *"Is it raining in London today?"*

Watch the agent reason step by step — tool calls, search results, and the final answer all stream live.

---

## Graph Visualization

```bash
python visualize_graph.py
```

Generates three outputs in `./output/`:

| File | Description |
|------|-------------|
| `react_graph.html` | Interactive Pyvis diagram — drag nodes, hover for descriptions |
| `react_graph.png` | Static Matplotlib render for docs/reports |
| `react_graph_mermaid.png` | LangGraph's native Mermaid render (requires internet) |

The HTML diagram opens automatically in your default browser.

```bash
# Custom output directory, skip auto-open
python visualize_graph.py --out-dir ./my_output --no-open
```

---

## Screenshots

### Agent UI — Dallas weather query
![Dallas query](screenshots/ui_dallas.png)

### Agent UI — Mumbai weather query
![Mumbai query](screenshots/ui_mumbai.png)

### Interactive graph (react_graph.html)
![Graph](screenshots/graph.png)

---

## How It Works

### `agent.py` — the logic layer

`build_agent(openai_key, tavily_key)` compiles and returns a LangGraph `CompiledStateGraph`. The graph has two nodes and conditional routing:

```python
workflow.add_node("agent", call_model)   # GPT-4o-mini with tools bound
workflow.add_node("tools", tool_node)    # executes all tool calls
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"continue": "tools", "end": END},
)
```

`stream_agent(graph, query)` is a generator that yields each message as a dict:

```python
for step in stream_agent(graph, "Weather in Tokyo?"):
    print(step)
# {"role": "ai", "content": "", "tool_calls": [{"name": "search_tool", "args": {...}}]}
# {"role": "tool", "content": "[...]", "name": "search_tool", "tool_calls": []}
# {"role": "ai", "content": "The weather in Tokyo is...", "tool_calls": []}
```

### `app.py` — the web layer

Three routes:

```
POST /configure   → initialize agent with API keys
GET  /query?q=... → stream agent steps as SSE events
GET  /status      → check if agent is initialized
```

### Tools

| Tool | Input | Logic |
|------|-------|-------|
| `search_tool` | `query: str` | Calls Tavily API for real-time web results |
| `recommend_clothing` | `weather: str` | Keyword matching → clothing advice string |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent framework | [LangGraph](https://github.com/langchain-ai/langgraph) 0.3.34 |
| LLM | OpenAI GPT-4o-mini via [LangChain-OpenAI](https://github.com/langchain-ai/langchain) |
| Web search | [Tavily](https://tavily.com/) via langchain-community |
| Web server | [Flask](https://flask.palletsprojects.com/) 3.x |
| Streaming | Server-Sent Events (SSE) |
| Frontend | Bootstrap 5.3 + IBM Plex fonts |
| Graph viz | [Pyvis](https://pyvis.readthedocs.io/) + Matplotlib |

---

## Requirements

```
flask>=3.0.0
langgraph==0.3.34
langchain==0.3.24
langchain-openai==0.3.14
langchain-community==0.3.23
langchainhub==0.1.21
tavily-python
pyvis>=0.3.2
matplotlib>=3.8.0
```

Python 3.10+ recommended.

---

## Based On

This project is a full conversion of the **"Build Reasoning and Acting AI Agents with LangGraph"** lab from [IBM Skills Network](https://skills.network), expanded into a production-ready web application with a streaming UI and graph visualization tooling.

---

## License

MIT
