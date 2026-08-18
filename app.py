"""
Simple Divi's Labs chatbot backend.

No Docker, no Qdrant, no local GPU needed. This calls Groq's free API,
which runs open-source models (Llama 3.1, etc.) for you in the cloud.

Deploy this on Render.com (free) - see DEPLOY_STEPS.md for exact clicks.
"""
import os
import json
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq

app = Flask(__name__)
CORS(app)  # allows your website to call this backend

# Set this in Render's dashboard as an environment variable (never put it in code)
client = Groq(api_key=os.environ["GROQ_API_KEY"])

DATA_PATH = Path(__file__).parent / "divislabs_content.json"
with open(DATA_PATH) as f:
    PAGES = json.load(f)

SYSTEM_PROMPT = """You are the official assistant for Divi's Labs (divislabs.com).
Answer ONLY using the information provided in the Context section below.
If the context does not contain the answer, say clearly that you don't have
that information and suggest the visitor contact Divi's Labs directly.
Never invent product names, certifications, or facts not present in the context.
Keep answers concise and factual, in 2-4 sentences."""


def find_relevant_pages(question: str, top_k: int = 3) -> list[dict]:
    """Simple keyword-overlap search. No database needed for a small
    knowledge base like this. Good enough for a few dozen pages."""
    question_words = set(question.lower().split())
    scored = []
    for page in PAGES:
        page_words = set((page["title"] + " " + page["content"]).lower().split())
        overlap = len(question_words & page_words)
        if overlap > 0:
            scored.append((overlap, page))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [page for _, page in scored[:top_k]]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400

    relevant = find_relevant_pages(question)

    if relevant:
        context = "\n\n".join(f"[{p['title']}]\n{p['content']}" for p in relevant)
    else:
        context = "(No relevant approved content was found for this question.)"

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",  # open-weight model, hosted by Groq for free
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0.3,
        max_tokens=300,
    )
    answer = completion.choices[0].message.content.strip()

    sources = [{"title": p["title"], "url": p["url"]} for p in relevant]

    return jsonify({"answer": answer, "sources": sources})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
