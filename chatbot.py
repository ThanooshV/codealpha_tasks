"""
Flexible FAQ Chatbot backend using TF-IDF + cosine similarity.
- Handles rephrased questions better.
- No NLTK required.
- Supports adding FAQs dynamically.
- CORS enabled for frontend HTML.

Install dependencies:
pip install flask scikit-learn flask-cors
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import threading

app = Flask(__name__)
CORS(app)  # Enable frontend access

# ---------- FAQ data ----------
faq_lock = threading.Lock()
_faqs = [
    {"question": "What is your return policy?", "answer": "You can return products within 30 days of purchase."},
    {"question": "How can I track my order?", "answer": "You can track your order using the tracking ID sent to your email."},
    {"question": "Do you offer international shipping?", "answer": "Yes, we ship to over 50 countries worldwide."},
    {"question": "What payment methods do you accept?", "answer": "We accept credit cards, debit cards, UPI, and PayPal."},
    {"question": "How can I contact customer support?", "answer": "You can email us at support@example.com or call 123-456-7890."},
    {"question": "Do you provide refunds?", "answer": "Refunds are processed within 7 business days after receiving the returned product."},
    {"question": "Can I change my shipping address?", "answer": "Yes, you can change your shipping address before the order is shipped."},
    {"question": "Do you offer discounts for students?", "answer": "Yes, we offer a 10% discount for students with a valid student ID."}
]

# ---------- Vectorizer ----------
vectorizer = TfidfVectorizer(stop_words='english')
_question_vectors = None

def rebuild_vector_index():
    global _question_vectors
    with faq_lock:
        questions = [faq["question"] for faq in _faqs]
        if not questions:
            _question_vectors = None
            return
        _question_vectors = vectorizer.fit_transform(questions)

# Build initial vector index
rebuild_vector_index()

# ---------- Helper: Find best answer ----------
def find_best_answer(user_text, threshold=0.2):
    """
    Returns (answer, score, question_index)
    If similarity below threshold, returns (None, score, index)
    """
    if not user_text or _question_vectors is None:
        return None, 0.0, None
    user_vec = vectorizer.transform([user_text])
    scores = cosine_similarity(user_vec, _question_vectors)[0]
    best_idx = int(scores.argmax())
    best_score = float(scores[best_idx])
    if best_score < threshold:
        return None, best_score, best_idx
    return _faqs[best_idx]["answer"], best_score, best_idx

# ---------- Routes ----------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    question = str(data.get("question", "")).strip()
    if not question:
        return jsonify({"error": "Empty question"}), 400
    answer, score, idx = find_best_answer(question, threshold=0.2)
    if answer is None:
        return jsonify({
            "reply": "Sorry, I couldn't understand your question. Try rephrasing it or ask another question.",
            "score": score,
            "best_match_index": idx
        })
    return jsonify({
        "reply": answer,
        "score": round(score, 4),
        "matched_question": _faqs[idx]["question"],
        "best_match_index": idx
    })

@app.route("/add_faq", methods=["POST"])
def add_faq():
    data = request.get_json(silent=True) or {}
    q = str(data.get("question", "")).strip()
    a = str(data.get("answer", "")).strip()
    if not q or not a:
        return jsonify({"error": "Both 'question' and 'answer' are required."}), 400
    with faq_lock:
        _faqs.append({"question": q, "answer": a})
        rebuild_vector_index()
    return jsonify({"status": "ok", "total_faqs": len(_faqs)}), 201

@app.route("/faqs", methods=["GET"])
def get_faqs():
    with faq_lock:
        return jsonify({"faqs": _faqs})

@app.route("/", methods=["GET"])
def index():
    return """
    <h3>Flexible FAQ Chatbot Backend</h3>
    <p>POST /chat with JSON {"question": "..."} to get answers.</p>
    <p>POST /add_faq to add more FAQs dynamically.</p>
    <p>GET /faqs to see current FAQs.</p>
    """

# ---------- Run ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
