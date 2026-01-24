import os
import base64
import json
import time
from flask import Flask, request, jsonify
from openai import OpenAI

# Serve static files from /public
# static_url_path="" means files are served at root like /quiz.html
app = Flask(__name__, static_folder="public", static_url_path="")

# OpenAI key must be set in Render Environment Variables: OPENAI_API_KEY
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    timeout=180
)

@app.route("/")
def index():
    # Serves /public/quiz.html
    return app.send_static_file("quiz.html")


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


def generate_pass(pdf_name, pdf_data_url, mcq_n, tf_n, fib_n, label):
    schema = strict_schema()

    instructions = f"""
Generate revision questions strictly from the PDF content ONLY.
Return ONLY JSON matching the provided schema. No extra fields.

This is {label} of 2. Create:
- MCQ: {mcq_n}
- True/False: {tf_n}
- Fill-in-the-Blank: {fib_n}

Rules:
- MCQ: Exactly 4 options A-D, only ONE correct.
- TF: answer must be "True" or "False".
- FIB: prompt must contain "__________" once per blank.
  Provide answers as an array of arrays (one array per blank) with acceptable variants.
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

    # Parse JSON from output_text (avoids output_parsed issues)
    return json.loads(resp.output_text)


@app.route("/api/generate-quiz-from-pdf", methods=["POST"])
def generate_quiz():
    try:
        if "pdf" not in request.files:
            return "No PDF uploaded (field name must be 'pdf')", 400

        pdf = request.files["pdf"]
        pdf_bytes = pdf.read()
        if not pdf_bytes:
            return "PDF appears empty", 400

        # PDF must be sent as a base64 data URL
        raw_b64 = base64.b64encode(pdf_bytes).decode("utf-8")
        pdf_data_url = "data:application/pdf;base64," + raw_b64

        # Two-pass generation: total 30/20/10
        targets = {"mcq": 30, "tf": 20, "fib": 10}
        half = {"mcq": 15, "tf": 10, "fib": 5}

        t0 = time.time()
        print("Generate request received. Starting PASS 1…", flush=True)
        q1 = generate_pass(pdf.filename or "upload.pdf", pdf_data_url, half["mcq"], half["tf"], half["fib"], "PASS 1")
        print("PASS 1 complete. Starting PASS 2…", flush=True)
        q2 = generate_pass(pdf.filename or "upload.pdf", pdf_data_url, half["mcq"], half["tf"], half["fib"], "PASS 2")
        print("PASS 2 complete. Merging…", flush=True)

        merged = {
            "mcq": dedupe_by_prompt((q1.get("mcq") or []) + (q2.get("mcq") or []))[:targets["mcq"]],
            "tf":  dedupe_by_prompt((q1.get("tf")  or []) + (q2.get("tf")  or []))[:targets["tf"]],
            "fib": dedupe_by_prompt((q1.get("fib") or []) + (q2.get("fib") or []))[:targets["fib"]],
        }

        dt = time.time() - t0
        print(f"Done in {dt:.1f}s -> {len(merged['mcq'])} MCQ, {len(merged['tf'])} TF, {len(merged['fib'])} FIB", flush=True)
        return jsonify(merged)

    except Exception as e:
        print("ERROR:", repr(e), flush=True)
        return f"Server error: {str(e)}", 500


# For local runs only. On Render with gunicorn, this block is ignored.
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    print(f"Local run: http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
