"""
Flask API Server for Harry Potter Chatbot
Save as: api_server.py
Run: python api_server.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import traceback

# Import your existing chatbot file
# Make sure harrybot_enhanced.py is in the same folder
import harrybot_enhanced as chatbot

app = Flask(__name__)
CORS(app)  # Allow browser/Flutter to call this API


def startup():
    """
    Initialize chatbot once when the API starts.
    Important: Flask debug reload can run twice, so we disable reloader below.
    """
    chatbot.initialize_chat_log()
    chatbot.DIALOG_ID = chatbot.generate_dialog_id()

    print("✨ API Server Starting...")
    print(f"🆔 Dialog ID: {chatbot.DIALOG_ID}")
    print(f"📚 Corpus lines: {len(chatbot.corpus)}")
    print(f"🔍 FAISS vectors: {chatbot.faiss_index.ntotal}")


startup()


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "service": "Harry Potter Chatbot API"
    })


@app.route("/api/chat", methods=["POST"])
def chat_endpoint():
    """
    POST JSON:
    { "question": "Who is Harry Potter?" }

    Returns JSON:
    { "answer": "...", "dialog_id": "...", "status": "success" }
    """
    try:
        data = request.get_json(silent=True) or {}

        question = (data.get("question") or "").strip()
        if not question:
            return jsonify({
                "status": "error",
                "error": "Missing 'question' field"
            }), 400

        answer = chatbot.chat(question, chatbot.API_KEY, chatbot.DIALOG_ID)

        return jsonify({
            "status": "success",
            "answer": answer,
            "dialog_id": chatbot.DIALOG_ID
        })

    except Exception as e:
        # Print full error in terminal for debugging
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@app.route("/api/reset", methods=["POST"])
def reset_endpoint():
    """
    Reset conversation history + new dialog id.
    """
    try:
        chatbot.conversation_history = []
        chatbot.DIALOG_ID = chatbot.generate_dialog_id()

        return jsonify({
            "status": "success",
            "message": "Conversation reset",
            "dialog_id": chatbot.DIALOG_ID
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@app.route("/api/stats", methods=["GET"])
def stats_endpoint():
    """
    Returns some debug stats.
    """
    try:
        return jsonify({
            "status": "success",
            "dialog_id": chatbot.DIALOG_ID,
            "corpus_lines": len(chatbot.corpus),
            "conversation_turns": len(chatbot.conversation_history),
            "faiss_vectors": chatbot.faiss_index.ntotal
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("⚡ HARRY POTTER CHATBOT API SERVER ⚡")
    print("=" * 70)
    print("\nEndpoints:")
    print("  POST /api/chat    - Send a question")
    print("  POST /api/reset   - Reset conversation")
    print("  GET  /api/stats   - Get statistics")
    print("  GET  /api/health  - Health check")
    print("\n" + "=" * 70)
    print("\n🚀 Starting server on http://localhost:5000\n")

    # IMPORTANT: use_reloader=False prevents double-loading model in debug
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
