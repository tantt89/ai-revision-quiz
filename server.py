import os
import base64
import json
from flask import Flask, request, jsonify
from openai import OpenAI

# -----------------------------
# Flask setup
# -----------------------------
app = Flask(__name__, static_folder="public", static_url_path="")

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    timeout=180
)

@app.route("/")
def index():
    # Serve public/quiz.html
    return app.send_static_file("quiz.html")


# -----------------------------
# STRICT MCQ-ONLY SCHEMA
# -----------------------------
def mcq_schema():
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["mcq"],
        "properties": {
            "mcq": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["prompt", "options", "answer"],
                    "properties": {
                        "prompt": {"type": "string"},
                        "options": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["A", "B", "C", "D"],
                            "properties": {
                                "A": {"type": "string"},
                                "B": {"type": "string"},
                                "C": {"type": "string"},
                                "D": {"type": "string"},
                            },
                        },
                        "answer": {
                            "type": "string",
                            "enum": ["A", "B", "C", "D"]
                        },
                    },
                },
            }
        },
    }


# -----------------------------
# Helpers
# -----------------------------
def normalize(text: str) -> str:
    return " ".join((text or "").lower().strip().split())


def dedupe_mcq(items):
    seen = set()
    out = []
    for q in items or []:
        key = normalize(q.get("prompt"))
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(q)
    return out


def generate_pass(pdf_name, pdf_data_url, count, label):
    schema = mcq_schema()

    instructions = f"""
Generate revision questions strictly from the PDF content ONLY.

Return ONLY JSON matching the provided schema.

This is {label} of 2.
Generate EXACTLY {count} Multiple Choice Questions (MCQ).

Rules:
- Each question must test a DIFFERENT concept or fact.
- Do NOT repeat or paraphrase earlier questions.
- Exactly 4 options (A–D).
- Only ONE correct answer.
- No explanations.
"""

    resp = client.responses.create(
        model="gpt-5",
        instructions=instructions,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "filename": pdf_name,
                        "file_data": pdf_data_url,
                    }
                ],
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "quiz",
                "schema": schema,
                "strict": True,
            }
        },
    )

    return json.loads(resp.output_text)


# -----------------------------
# API
# -----------------------------
@app.route("/api/generate-quiz-from-pdf", methods=["POST"])
def generate_quiz():
    try:
        if "pdf" not in request.files:
            return "No PDF uploaded", 400

        pdf = request.files["pdf"]
        data = pdf.read()
        if not data:
            return "Empty PDF", 400

        # Convert PDF → base64 data URL
        pdf_b64 = base64.b64encode(data).decode("utf-8")
        pdf_data_url = "data:application/pdf;base64," + pdf_b64

        print("PASS 1: generating 25 MCQ…", flush=True)
        q1 = generate_pass(pdf.filename or "upload.pdf", pdf_data_url, 25, "PASS 1")

        print("PASS 2: generating 25 MCQ…", flush=True)
        q2 = generate_pass(pdf.filename or "upload.pdf", pdf_data_url, 25, "PASS 2")

        # Merge + dedupe
        mcq = dedupe_mcq((q1.get("mcq") or []) + (q2.get("mcq") or []))
        mcq = mcq[:50]

        print(f"Returning {len(mcq)} MCQ", flush=True)
        return jsonify({"mcq": mcq})

    except Exception as e:
        print("ERROR:", repr(e), flush=True)
        return jsonify({"error": str(e)}), 500


# -----------------------------
# Local run (ignored by gunicorn)
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
