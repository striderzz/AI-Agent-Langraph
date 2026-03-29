"""
ReAct Agent Web Application
Flask + Bootstrap UI for the LangGraph ReAct agent
"""

import json
from flask import Flask, render_template, request, Response, stream_with_context, jsonify
from agent import build_agent, stream_agent

app = Flask(__name__)

# Global agent instance (initialized on first use or via /configure)
_agent = None


def get_agent():
    return _agent


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/configure", methods=["POST"])
def configure():
    global _agent
    data = request.json
    openai_key = data.get("openai_key", "").strip()
    tavily_key = data.get("tavily_key", "").strip()

    if not openai_key or not tavily_key:
        return jsonify({"success": False, "error": "Both API keys are required."}), 400

    try:
        _agent = build_agent(openai_key, tavily_key)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/query", methods=["GET"])
def query():
    q = request.args.get("q", "").strip()
    if not q:
        return Response("data: " + json.dumps({"error": "Empty query"}) + "\n\n",
                        mimetype="text/event-stream")

    agent = get_agent()
    if agent is None:
        return Response("data: " + json.dumps({"error": "Agent not configured. Add your API keys first."}) + "\n\n",
                        mimetype="text/event-stream")

    def generate():
        try:
            for step in stream_agent(agent, q):
                yield f"data: {json.dumps(step)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/status")
def status():
    return jsonify({"configured": _agent is not None})


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)
