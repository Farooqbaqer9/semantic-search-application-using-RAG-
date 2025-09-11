from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "Flask API for Semantic Search RAG Application (Gemini)"

if __name__ == "__main__":
    app.run(debug=True)
