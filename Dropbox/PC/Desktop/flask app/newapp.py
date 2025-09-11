
from flask import Flask, request, jsonify, render_template_string
from transformers import pipeline

app = Flask(__name__)

# Load HuggingFace model (example: question answering)
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Simple homepage with HTML form
@app.route("/", methods=["GET"])
def home():
    return render_template_string("""
    <html>
    <head>
        <title>🚀 Flask + HuggingFace Demo</title>
    </head>
    <body style="font-family: Arial; margin: 40px;">
        <h2>🚀 Flask + HuggingFace Demo</h2>
        <p>Type your query below:</p>
        <form action="/search" method="post">
            <label><b>Question:</b></label><br>
            <input type="text" name="query" style="width:300px;" required><br><br>
            <label><b>Context (optional):</b></label><br>
            <textarea name="context" rows="5" cols="50">Flask is a lightweight WSGI web application framework in Python. It is designed to make getting started quick and easy, with the ability to scale up to complex applications.</textarea><br><br>
            <button type="submit">Search</button>
        </form>
    </body>
    </html>
    """)

# Handle search (works with both JSON and Form)
@app.route("/search", methods=["POST"])
def search():
    if request.is_json:
        data = request.get_json()
        query = data.get("query", "")
        context = data.get("context", "Flask is a web framework in Python.")
    else:
        query = request.form.get("query", "")
        context = request.form.get("context", "Flask is a web framework in Python.")

    if not query:
        return jsonify({"error": "No query provided"}), 400

    # HuggingFace model answer
    result = qa_pipeline(question=query, context=context)

    # If form request → render result nicely
    if not request.is_json:
        return render_template_string("""
        <h2>🔍 Search Result</h2>
        <p><b>Question:</b> {{ query }}</p>
        <p><b>Answer:</b> {{ answer }}</p>
        <a href="/">⬅ Back</a>
        """, query=query, answer=result["answer"])

    # Otherwise return JSON
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)




