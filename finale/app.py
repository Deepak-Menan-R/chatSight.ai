from flask import Flask, request, jsonify
from dotenv import load_dotenv
from master_agent.controller import MasterAgent

load_dotenv()
app = Flask(__name__)

master_agent = MasterAgent()

@app.route("/api/v1/master", methods=["POST"])
def master_controller():
    data = request.get_json()
    task = data.get("task")
    if not task:
        return jsonify({"error": "Task is required"}), 400

    response = master_agent.handle_task(task)
    return jsonify({"response": response})

@app.route("/", methods=["GET"])
def home():
    return "LangChain Multi-Agent Flask API ✅"

if __name__ == "__main__":
    app.run(debug=True, port=5000)
