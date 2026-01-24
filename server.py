from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI
import base64
import os
import json
import time

# Serve static quiz.html from /public
app = Flask(__name__, static_folder="public")

# Read API key from environment (Render: Environment Variables; Local: setx / export)
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=180  # seconds per API call
)


@app.route("/")
def home():
    # Serves /public/quiz.html
    return send_from_directory("public", "quiz.html")


def build_schema():
    # Strict JSON Schema: additionalProperties must be False for all objects
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


def dedupe_by_prompt(items):
    seen = set()
    out = []
    for it in items or []:
        p = (it.get("prompt") or "").strip().lower()
        if not p:
            continue
        if p in seen:
            continue
        seen.add(p)
        out.append(it)
    return out


def generate_pass(pdf_filename, pdf_data_url, mcq_n, tf_n, fib_n, pass_label):
    schema = build_schema()

    instructions = f"""
Generate revision questions strictly from the PDF content ONLY.

Return ONLY JSON matching the provided schema. No extra fields.

This is {pass_label} of 2. Create:
- MCQ: {mcq_n}
- True/False: {tf_n}
- Fill-in-the-Blank: {fib_n}

Rules:
- MCQ: Exactly 4 options A-D, only ONE correct.
- TF: answer must be "True" or "False".
- FIB: prompt must contain "__________" once per blank.
  Provide answers as an array of arrays (one array per blank) with acceptable variants.
- Avoid repeating earlier questions; aim for variety across topics in the PDF.
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
                        "filename": pdf_filename,
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

    # Parse JSON from output_text (robust across SDK versions)
    return json.loads(resp.output_text)


@app.route("/api/generate-quiz-from-pdf", methods=["POST"])
def generate_quiz():
    try:
        if "pdf" not in request.files:
            return "No PDF uploaded (field name must be 'pdf')", 400

        pdf = request.files["pdf"]
        if not pdf.filename.lower().endswith(".pdf"):
            return "Uploaded file is not a PDF", 400

        pdf_bytes = pdf.read()
        if not pdf_bytes:
            return "PDF appears empty", 400

        # Convert to base64 Data URL (required format for PDF file input)
        raw_b64 = base64.b64encode(pdf_bytes).decode("utf-8")
        pdf_data_url = "data:application/pdf;base64," + raw_b64

        # Targets
        targets = {"mcq": 30, "tf": 20, "fib": 10}
        half = {"mcq": 15, "tf": 10, "fib": 5}

        t0 = time.time()
        print("POST /api/generate-quiz-from-pdf: starting pass 1…")
        q1 = generate_pass(pdf.filename, pdf_data_url, half["mcq"], half["tf"], half["fib"], "PASS 1")
        print("Pass 1 complete.")

        print("POST /api/generate-quiz-from-pdf: starting pass 2…")
        q2 = generate_pass(pdf.filename, pdf_data_url, half["mcq"], half["tf"], half["fib"], "PASS 2")
        print("Pass 2 complete.")

        merged = {
            "mcq": dedupe_by_prompt((q1.get("mcq") or []) + (q2.get("mcq") or [])),
            "tf":  dedupe_by_prompt((q1.get("tf") or []) + (q2.get("tf") or [])),
            "fib": dedupe_by_prompt((q1.get("fib") or []) + (q2.get("fib") or [])),
        }

        # Trim to exact counts
        merged["mcq"] = merged["mcq"][:targets["mcq"]]
        merged["tf"]  = merged["tf"][:targets["tf"]]
        merged["fib"] = merged["fib"][:targets["fib"]]

        dt = time.time() - t0
        print(f"POST done in {dt:.1f}s -> {len(merged['mcq'])} MCQ, {len(merged['tf'])} TF, {len(merged['fib'])} FIB")

        return jsonify(merged)

    except Exception as e:
        print("ERROR:", repr(e))
        return f"Server error: {str(e)}", 500


if __name__ == "__main__":
    # Render sets PORT dynamically; locally it will use 3000
    port = int(os.environ.get("PORT", 3000))
    print(f"Starting Flask on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)
