import os
import base64
import json
import time
from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI

# -----------------------------
# App setup
# -----------------------------
app = Flask(__name__, static_folder="public")

# OpenAI client (key comes from Render Environment Variables)
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    timeout=180
)

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def index():
    return send_from_directory("public", "quiz.html")


def strict_schema():
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["mcq", "tf", "fib"],
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
                        "answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
                    },
                },
            },
            "tf": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["prompt", "answer"],
                    "properties": {
                        "prompt": {"type": "string"},
                        "answer": {"type": "string", "enum": ["True", "False"]},
                    },
                },
            },
            "fib": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["prompt", "answers"],
                    "properties": {
                        "prompt": {"type": "string"},
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
    }


def generate_pass(pdf_name, pdf_data_url, mcq, tf, fib, label):
    schema = strict_schema()

    instructions = f"""
Generate revision questions strictly from the PDF content ONLY.

Return ONLY JSON matching the provided schema.

This is {label} of 2.
MCQ: {mcq}
True/False: {tf}
Fill-in-the-Blank: {fib}

Rules:
- MCQ: 4 options A-D, one correct
- TF: answer must be "True" or "False"
- FIB: use "__________" and provide acceptable answers
- No explanations
"""

    r = client.responses.create(
        model="gpt-5",
        instructions=instructions,
        input=[{
            "role": "user",
            "content": [{
                "type": "input_file",
                "filename": pdf_name,
                "file_data": pdf_data_url
            }]
        }],
        text={
            "format": {
                "type": "json_schema",
                "name": "quiz",
                "schema": schema,
                "strict": True
            }
        }
    )

    return json.loads(r.output_text)


@app.route("/api/generate-quiz-from-pdf", methods=["POST"])
def generate():
    if "pdf" not in request.files:
        return "No PDF uploaded", 400

    pdf = request.files["pdf"]
    data = pdf.read()

    raw = base64.b64encode(data).decode("utf-8")
    pdf_data_url = "data:application/pdf;base64," + raw

    print("Starting AI generation…")

    q1 = generate_pass(pdf.filename, pdf_data_url, 15, 10, 5, "PASS 1")
    q2 = generate_pass(pdf.filename, pdf_data_url, 15, 10, 5, "PASS 2")

    def dedupe(items):
        seen = set()
        out = []
        for x in items:
            p = x["prompt"].lower().strip()
            if p not in seen:
                seen.add(p)
                out.append(x)
        return out

    result = {
        "mcq": dedupe(q1["mcq"] + q2["mcq"])[:30],
        "tf":  dedupe(q1["tf"]  + q2["tf"])[:20],
        "fib": dedupe(q1["fib"] + q2["fib"])[:10],
    }

    print("Generation complete.")
    return jsonify(result)


# -----------------------------
# ENTRY POINT (Render-safe)
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ["PORT"])  # Render ALWAYS provides this
    print(f"Listening on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
