from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI
import base64
import os
import json
import time

app = Flask(__name__, static_folder="public")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=180)  # 3 min timeout per call


@app.route("/")
def home():
    return send_from_directory("public", "quiz.html")


def build_schema():
    # Strict JSON Schema: additionalProperties False everywhere
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


def call_openai_for_quiz(pdf_filename: str, pdf_data_url: str, mcq: int, tf: int, fib: int, pass_label: str):
    schema = build_schema()

    instructions = f"""
Generate revision questions strictly from the PDF content ONLY.

Return ONLY JSON matching the provided schema. No extra fields.

This is {pass_label} of 2. Create:
- MCQ: {mcq}
- True/False: {tf}
- Fill-in-the-Blank: {fib}

Rules:
- MCQ: Exactly 4 options A-D, only ONE correct.
- TF: answer must be "True" or "False".
- FIB: prompt must contain "__________" once per blank.
  Provide answers as an array of arrays (one array per blank) with acceptable variants.
- Avoid repeating questions from earlier pass. Aim for variety across topics in the PDF.
- No explanations.
"""

    response = client.responses.create(
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

    # Parse JSON from output_text
    return json.loads(response.output_text)


def dedupe_by_prompt(items):
    seen = set()
    out = []
    for it in items:
        p = (it.get("prompt") or "").strip().lower()
        if not p:
            continue
        if p in seen:
            continue
        seen.add(p)
        out.append(it)
    return out


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

        # Convert PDF to base64 DATA URL
        raw_b64 = base64.b64encode(pdf_bytes).decode("utf-8")
        pdf_data_url = "data:application/pdf;base64," + raw_b64

        # Two-pass generation totals: (15/10/5) + (15/10/5) = (30/20/10)
        targets = {"mcq": 30, "tf": 20, "fib": 10}
        pass_counts = {"mcq": 15, "tf": 10, "fib": 5}

        t0 = time.time()
        print("Starting generation pass 1…")
        q1 = call_openai_for_quiz(pdf.filename, pdf_data_url,
                                  pass_counts["mcq"], pass_counts["tf"], pass_counts["fib"],
                                  "PASS 1")
        print("Pass 1 done.")

        print("Starting generation pass 2…")
        q2 = call_openai_for_quiz(pdf.filename, pdf_data_url,
                                  pass_counts["mcq"], pass_counts["tf"], pass_counts["fib"],
                                  "PASS 2")
        print("Pass 2 done.")

        merged = {
            "mcq": dedupe_by_prompt((q1.get("mcq") or []) + (q2.get("mcq") or [])),
            "tf":  dedupe_by_prompt((q1.get("tf") or []) + (q2.get("tf") or [])),
            "fib": dedupe_by_prompt((q1.get("fib") or []) + (q2.get("fib") or [])),
        }

        # Trim to exact targets (in case dedupe left extra)
        merged["mcq"] = merged["mcq"][:targets["mcq"]]
        merged["tf"]  = merged["tf"][:targets["tf"]]
        merged["fib"] = merged["fib"][:targets["fib"]]

        dt = time.time() - t0
        print(f"Done. Returned {len(merged['mcq'])} MCQ, {len(merged['tf'])} TF, {len(merged['fib'])} FIB in {dt:.1f}s")

        return jsonify(merged)

    except Exception as e:
        print("ERROR:", repr(e))
        return f"Server error: {str(e)}", 500


if __name__ == "__main__":
    print("Open http://localhost:3000")
    app.run(port=3000)
