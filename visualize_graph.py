"""
visualize_graph.py
==================
Standalone script to render the ReAct LangGraph state-machine as an
interactive HTML diagram using Pyvis AND as a static PNG using Matplotlib.

Usage
-----
    python visualize_graph.py

Outputs
-------
    react_graph.html  — interactive, browser-openable Pyvis network
    react_graph.png   — static matplotlib diagram
"""

import os
import sys
import json
import textwrap
import webbrowser

# --------------------------------------------------------------------------- #
# Graph definition (mirrors agent.py — no API keys needed)                    #
# --------------------------------------------------------------------------- #
try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    from typing import Annotated, Sequence, TypedDict
    from langchain_core.messages import BaseMessage

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]

    def _dummy_call_model(state):
        return state

    def _dummy_tool_node(state):
        return state

    def _should_continue(state):
        return "end"

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", _dummy_call_model)
    workflow.add_node("tools", _dummy_tool_node)
    workflow.add_edge("tools", "agent")
    workflow.add_conditional_edges(
        "agent",
        _should_continue,
        {"continue": "tools", "end": END},
    )
    workflow.set_entry_point("agent")
    graph = workflow.compile()
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    graph = None


# --------------------------------------------------------------------------- #
# Graph data (used whether or not LangGraph is installed)                     #
# --------------------------------------------------------------------------- #

NODES = [
    {
        "id": "__start__",
        "label": "START",
        "color": "#3fb950",
        "shape": "diamond",
        "desc": "Entry point.\nEvery query begins here.",
        "size": 22,
    },
    {
        "id": "agent",
        "label": "Agent\n(call_model)",
        "color": "#58a6ff",
        "shape": "box",
        "desc": "Invokes GPT-4o-mini with the\nfull conversation history.\nMay emit tool calls.",
        "size": 30,
    },
    {
        "id": "tools",
        "label": "Tools\n(tool_node)",
        "color": "#d2a679",
        "shape": "box",
        "desc": "Executes tool calls:\n• search_tool (Tavily)\n• recommend_clothing",
        "size": 28,
    },
    {
        "id": "__end__",
        "label": "END",
        "color": "#f85149",
        "shape": "diamond",
        "desc": "Final response delivered\nto the user.",
        "size": 22,
    },
]

EDGES = [
    ("__start__", "agent",  ""),
    ("agent",     "tools",  "tool_calls\npresent"),
    ("agent",     "__end__","no tool calls\n(done)"),
    ("tools",     "agent",  "results\nappended"),
]


# --------------------------------------------------------------------------- #
# 1. Matplotlib / static PNG                                                   #
# --------------------------------------------------------------------------- #

def save_matplotlib(output_path="react_graph.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    except ImportError:
        print("[visualize_graph] matplotlib not found — skipping PNG output.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Node positions  (x, y)
    pos = {
        "__start__": (1.0, 3.0),
        "agent":     (4.0, 3.0),
        "tools":     (4.0, 1.0),
        "__end__":   (8.0, 3.0),
    }

    box_w, box_h = 1.8, 0.85
    colors = {n["id"]: n["color"] for n in NODES}
    shapes = {n["id"]: n["shape"] for n in NODES}
    descs  = {n["id"]: n["desc"]  for n in NODES}
    labels = {n["id"]: n["label"] for n in NODES}

    # Draw nodes
    for nid, (x, y) in pos.items():
        c = colors[nid]
        bx = x - box_w/2
        by = y - box_h/2
        if shapes[nid] == "diamond":
            # diamond polygon
            dx, dy = box_w/2, box_h/2
            diamond = plt.Polygon(
                [[x, y+dy], [x+dx*1.2, y], [x, y-dy], [x-dx*1.2, y]],
                closed=True, facecolor=c+"30", edgecolor=c, linewidth=2, zorder=3,
            )
            ax.add_patch(diamond)
        else:
            rect = FancyBboxPatch(
                (bx, by), box_w, box_h,
                boxstyle="round,pad=0.05",
                facecolor=c+"20", edgecolor=c, linewidth=2, zorder=3,
            )
            ax.add_patch(rect)

        # Label
        lbl = labels[nid].replace("\n", "\n")
        ax.text(x, y, lbl, ha="center", va="center",
                fontsize=9, fontweight="bold", color=c,
                fontfamily="monospace", zorder=4,
                multialignment="center")

        # Description below
        ax.text(x, y - box_h/2 - 0.28,
                descs[nid],
                ha="center", va="top",
                fontsize=6.5, color="#7d8590",
                fontfamily="monospace",
                multialignment="center", zorder=4)

    # Draw edges
    edge_style = dict(arrowstyle="-|>", color="#58a6ff",
                      lw=1.5, mutation_scale=14,
                      connectionstyle="arc3,rad=0")

    def draw_edge(src, dst, label, rad=0.0):
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        # offset arrow ends to stop at box boundary
        dx, dy = x1-x0, y1-y0
        length = (dx**2+dy**2)**.5
        if length == 0: return
        ux, uy = dx/length, dy/length
        pad_src = box_w/2 + 0.05
        pad_dst = box_w/2 + 0.08
        xs = x0 + ux*pad_src
        ys = y0 + uy*pad_src
        xe = x1 - ux*pad_dst
        ye = y1 - uy*pad_dst

        cs = f"arc3,rad={rad}"
        arr = FancyArrowPatch(
            (xs, ys), (xe, ye),
            arrowstyle="-|>",
            color="#58a6ff",
            linewidth=1.5,
            mutation_scale=12,
            connectionstyle=cs,
            zorder=2,
        )
        ax.add_patch(arr)

        # edge label at midpoint
        if label:
            mx = (xs+xe)/2 + (-0.25 if rad < 0 else 0.25 if rad > 0 else 0)
            my = (ys+ye)/2 + (0.2 if rad != 0 else 0)
            ax.text(mx, my, label.replace("\n"," "),
                    fontsize=6.5, color="#d2a679",
                    ha="center", va="center",
                    fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.1", fc="#0d1117", ec="none"),
                    zorder=5)

    draw_edge("__start__", "agent",   "",                      rad=0.0)
    draw_edge("agent",     "__end__", "no tool calls (done)",  rad=0.0)
    draw_edge("agent",     "tools",   "tool calls present",    rad=0.2)
    draw_edge("tools",     "agent",   "results appended",      rad=0.2)

    # Title
    ax.set_title("ReAct Agent — LangGraph State Machine",
                 fontsize=13, fontweight="bold", color="#e6edf3",
                 fontfamily="monospace", pad=12)

    # Legend
    legend_items = [
        mpatches.Patch(facecolor="#58a6ff30", edgecolor="#58a6ff", label="Agent node (reasoning)"),
        mpatches.Patch(facecolor="#d2a67930", edgecolor="#d2a679", label="Tool node (execution)"),
        mpatches.Patch(facecolor="#3fb95030", edgecolor="#3fb950", label="Start / End"),
    ]
    ax.legend(handles=legend_items, loc="upper right",
              facecolor="#161b22", edgecolor="#30363d",
              labelcolor="#e6edf3", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor="#0d1117")
    plt.close()
    print(f"[visualize_graph] Static PNG saved → {output_path}")


# --------------------------------------------------------------------------- #
# 2. Pyvis / interactive HTML                                                  #
# --------------------------------------------------------------------------- #

def save_pyvis(output_path="react_graph.html"):
    try:
        from pyvis.network import Network
    except ImportError:
        print("[visualize_graph] pyvis not found — falling back to plain HTML output.")
        _save_plain_html(output_path)
        return

    net = Network(
        height="650px", width="100%",
        bgcolor="#0d1117", font_color="#e6edf3",
        directed=True,
        notebook=False,
    )

    pos_map = {
        "__start__": (100, 300),
        "agent":     (400, 300),
        "tools":     (400, 550),
        "__end__":   (700, 300),
    }

    for n in NODES:
        x, y = pos_map[n["id"]]
        net.add_node(
            n["id"],
            label=n["label"],
            title=n["desc"],
            color={"background": n["color"]+"25",
                   "border":     n["color"],
                   "highlight":  {"background": n["color"]+"50", "border": n["color"]}},
            shape="box" if n["shape"] == "box" else "diamond",
            size=n["size"],
            font={"color": n["color"], "size": 14, "face": "IBM Plex Mono"},
            borderWidth=2,
            x=x, y=y,
            physics=False,
        )

    edge_colors = {
        ("__start__", "agent"): "#3fb950",
        ("agent",  "tools"):    "#d2a679",
        ("agent",  "__end__"):  "#58a6ff",
        ("tools",  "agent"):    "#bc8cff",
    }

    for src, dst, lbl in EDGES:
        color = edge_colors.get((src, dst), "#58a6ff")
        net.add_edge(
            src, dst,
            label=lbl.replace("\n", " "),
            color={"color": color, "highlight": color},
            arrows={"to": {"enabled": True, "scaleFactor": 1.2}},
            font={"color": "#d2a679", "size": 11, "face": "IBM Plex Mono",
                  "background": "#161b22"},
            width=2,
            smooth={"type": "curvedCW", "roundness": 0.2} if src == "tools" else {},
        )

    net.set_options(json.dumps({
        "physics": {"enabled": False},
        "interaction": {
            "hover": True,
            "tooltipDelay": 100,
            "zoomView": True,
        },
        "edges": {"font": {"strokeWidth": 0}},
    }))

    # Inject custom CSS into the HTML
    raw = net.generate_html()
    custom = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600&display=swap');
body { background:#0d1117!important; margin:0; }
#mynetwork { border:1px solid #30363d!important; border-radius:12px; }
</style>
<div style="padding:18px 24px;background:#161b22;border-bottom:1px solid #30363d;
            font-family:'IBM Plex Mono',monospace;">
  <span style="color:#58a6ff;font-size:15px;font-weight:600;">
    ⬡ ReAct Agent — LangGraph State Machine
  </span>
  <span style="color:#7d8590;font-size:12px;margin-left:16px;">
    Hover nodes for details · Drag to explore
  </span>
</div>
"""
    raw = raw.replace("<body>", "<body>" + custom, 1)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(raw)

    print(f"[visualize_graph] Interactive HTML saved → {output_path}")


def _save_plain_html(output_path):
    """Fallback: pure HTML/CSS/JS diagram when pyvis is absent."""
    html = textwrap.dedent("""\
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8"/>
    <title>ReAct Agent Graph</title>
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&display=swap" rel="stylesheet"/>
    <style>
      body{background:#0d1117;color:#e6edf3;font-family:'IBM Plex Mono',monospace;margin:0;padding:0;}
      header{background:#161b22;border-bottom:1px solid #30363d;padding:16px 24px;}
      header h2{margin:0;color:#58a6ff;font-size:15px;}
      header small{color:#7d8590;font-size:11px;}
      svg{display:block;margin:40px auto;}
    </style>
    </head>
    <body>
    <header>
      <h2>⬡ ReAct Agent — LangGraph State Machine</h2>
      <small>Static fallback (install pyvis for interactive version)</small>
    </header>
    <svg viewBox="0 0 800 500" width="800" height="500" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L0,6 L8,3 z" fill="#58a6ff"/>
        </marker>
        <marker id="arr2" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L0,6 L8,3 z" fill="#d2a679"/>
        </marker>
        <marker id="arr3" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L0,6 L8,3 z" fill="#bc8cff"/>
        </marker>
      </defs>

      <!-- Edges -->
      <!-- start -> agent -->
      <line x1="110" y1="200" x2="280" y2="200" stroke="#3fb950" stroke-width="2" marker-end="url(#arr)"/>
      <!-- agent -> end -->
      <line x1="480" y1="200" x2="650" y2="200" stroke="#58a6ff" stroke-width="2" marker-end="url(#arr)"/>
      <!-- agent -> tools -->
      <path d="M 380 240 Q 380 320 380 340" stroke="#d2a679" stroke-width="2" fill="none" marker-end="url(#arr2)"/>
      <!-- tools -> agent -->
      <path d="M 340 340 Q 300 290 300 240" stroke="#bc8cff" stroke-width="2" fill="none" marker-end="url(#arr3)"/>

      <!-- Edge labels -->
      <text x="175" y="190" font-size="10" fill="#3fb950" text-anchor="middle">entry</text>
      <text x="565" y="190" font-size="10" fill="#58a6ff" text-anchor="middle">no tool calls</text>
      <text x="405" y="295" font-size="10" fill="#d2a679" text-anchor="start">tool calls</text>
      <text x="270" y="295" font-size="10" fill="#bc8cff" text-anchor="end">results back</text>

      <!-- START -->
      <polygon points="80,200 110,175 140,200 110,225" fill="#3fb95025" stroke="#3fb950" stroke-width="2"/>
      <text x="110" y="204" font-size="11" font-weight="bold" fill="#3fb950" text-anchor="middle">START</text>

      <!-- AGENT -->
      <rect x="280" y="165" width="200" height="70" rx="8" fill="#58a6ff20" stroke="#58a6ff" stroke-width="2"/>
      <text x="380" y="193" font-size="12" font-weight="bold" fill="#58a6ff" text-anchor="middle">Agent</text>
      <text x="380" y="210" font-size="10" fill="#58a6ff" text-anchor="middle">(call_model)</text>
      <text x="380" y="226" font-size="9" fill="#7d8590" text-anchor="middle">Reasons · Calls tools</text>

      <!-- TOOLS -->
      <rect x="280" y="340" width="200" height="70" rx="8" fill="#d2a67920" stroke="#d2a679" stroke-width="2"/>
      <text x="380" y="368" font-size="12" font-weight="bold" fill="#d2a679" text-anchor="middle">Tools</text>
      <text x="380" y="385" font-size="10" fill="#d2a679" text-anchor="middle">(tool_node)</text>
      <text x="380" y="401" font-size="9" fill="#7d8590" text-anchor="middle">search_tool · recommend_clothing</text>

      <!-- END -->
      <polygon points="660,200 690,175 720,200 690,225" fill="#f8514925" stroke="#f85149" stroke-width="2"/>
      <text x="690" y="204" font-size="11" font-weight="bold" fill="#f85149" text-anchor="middle">END</text>
    </svg>
    </body></html>
    """)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[visualize_graph] Fallback HTML saved → {output_path}")


# --------------------------------------------------------------------------- #
# 3. Optional: export LangGraph's own Mermaid PNG                              #
# --------------------------------------------------------------------------- #

def save_mermaid_png(graph, output_path="react_graph_mermaid.png"):
    if graph is None:
        print("[visualize_graph] LangGraph not available — skipping Mermaid PNG.")
        return
    try:
        img_bytes = graph.get_graph().draw_mermaid_png()
        with open(output_path, "wb") as f:
            f.write(img_bytes)
        print(f"[visualize_graph] Mermaid PNG saved → {output_path}")
    except Exception as e:
        print(f"[visualize_graph] Mermaid PNG skipped ({e})")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize the ReAct LangGraph agent.")
    parser.add_argument("--no-open", action="store_true",
                        help="Do not auto-open the HTML in a browser")
    parser.add_argument("--out-dir", default=".", help="Directory for output files")
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    html_path = os.path.join(out_dir, "react_graph.html")
    png_path  = os.path.join(out_dir, "react_graph.png")
    mm_path   = os.path.join(out_dir, "react_graph_mermaid.png")

    print("\n=== ReAct Agent Graph Visualizer ===\n")

    save_matplotlib(png_path)
    save_pyvis(html_path)
    save_mermaid_png(graph, mm_path)

    print("\nDone.")
    if not args.no_open:
        try:
            webbrowser.open("file://" + os.path.abspath(html_path))
        except Exception:
            pass
