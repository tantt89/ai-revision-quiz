import os
import base64
import json
from flask import Flask, request, jsonify
from openai import OpenAI

# Serve static files from /public
# With static_url_path="", /quiz.html maps to public/quiz.html
app = Flask(__name__, static_folder="public", static_url_path="")

# OPENAI_API_KEY must be set in Render Environment Variables (or locally)
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    timeout=180
)

@app.route("/")
def index():
    # Serve the quiz UI
    return app.send_static_file("quiz.html")


def schema_mcq_only():
    # Strict schema: object must be closed (additionalProperties=False everywhere)
    # We keep tf/fib fields but return them as empty arrays.
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
            "tf": {"type": "array"},
            "fib": {"type": "array"},
        },
    }


def normalize_prompt(p: str) -> str:
    # Stronger dedupe: ignore case + extra spaces
    return " ".join((p or "").strip().lower().split())


def dedupe_mcq(items):
    seen = set()
    out = []
    for it in items or []:
        p = normalize_prompt(it.get("prompt", ""))
        if not p:
            continue
        if p in seen:
            continue
        seen.add(p)
        out.append(it)
    return out


def generate_pass(pdf_filename: str, pdf_data_url: str, mcq_count: int, label: str):
    schema = schema_mcq_only()

    instructions = f"""
Generate revision questions strictly from the PDF content ONLY.

Return ONLY JSON matching the provided schema. No extra fields.

This is {label} of 2.
Create ONLY Multiple Choice Questions (MCQ): {mcq_count}
Create 0 True/False and 0 Fill-in-the-Blank.

Rules:
- Each question MUST test a different concept or fact from the PDF.
- Do NOT repeat or paraphrase earlier questions; keep all questions distinct.
- MCQ: Exactly 4 options (A–D), only ONE correct.
- No explanations.

Output requirements:
- "mcq" must contain exactly {mcq_count} items (if possible).
- "tf" must be [].
- "fib" must be [].
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

        # Convert to base64 data URL required for PDF file input
        raw_b64 = base64.b64encode(pdf_bytes).decode("utf-8")
        pdf_data_url = "data:application/pdf;base64," + raw_b64

        # 50 MCQ total, split into 2 passes of 25 to reduce timeouts
        print("Generate request received. Starting PASS 1 (25 MCQ)…", flush=True)
        q1 = generate_pass(pdf.filename or "upload.pdf", pdf_data_url, 25, "PASS 1")

        print("PASS 1 complete. Starting PASS 2 (25 MCQ)…", flush=True)
        q2 = generate_pass(pdf.filename or "upload.pdf", pdf_data_url, 25, "PASS 2")

        # Merge + dedupe
        merged_mcq = dedupe_mcq((q1.get("mcq") or []) + (q2.get("mcq") or []))

        # If dedupe reduced count < 50, we still return what we have (best effort)
        merged_mcq = merged_mcq[:50]

        result = {
            "mcq": merged_mcq,
            "tf": [],
            "fib": [],
        }

        print(f"Done. Returning {len(result['mcq'])} MCQ.", flush=True)
        return jsonify(result)

    except Exception as e:
        print("ERROR:", repr(e), flush=True)
        return f"Server error: {str(e)}", 500


# Local run only. Gunicorn ignores this block.
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    print(f"Local run: http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
