import os
import json
import time
from io import BytesIO

from flask import Flask, request, jsonify
from openai import OpenAI
from pypdf import PdfReader

app = Flask(__name__, static_folder="public", static_url_path="")

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    timeout=180
)

@app.route("/")
def index():
    return app.send_static_file("quiz.html")


# -----------------------------
# Per-user sessions (in memory)
# -----------------------------
SESSIONS = {}  # session_id -> {"pdf_hash": int, "mcq": [...], "updated": float}
MAX_SESSIONS = 200
SESSION_TTL_SECONDS = 6 * 60 * 60  # 6 hours


def cleanup_sessions():
    now = time.time()
    expired = [sid for sid, s in SESSIONS.items() if now - s.get("updated", now) > SESSION_TTL_SECONDS]
    for sid in expired:
        SESSIONS.pop(sid, None)

    if len(SESSIONS) > MAX_SESSIONS:
        oldest = sorted(SESSIONS.items(), key=lambda kv: kv[1].get("updated", 0))[: len(SESSIONS) - MAX_SESSIONS]
        for sid, _ in oldest:
            SESSIONS.pop(sid, None)


# -----------------------------
# MCQ-only strict schema
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
                        "answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
                    },
                },
            }
        },
    }


def norm(text: str) -> str:
    return " ".join((text or "").lower().strip().split())


def dedupe(existing_mcq, new_mcq):
    seen = {norm(q.get("prompt", "")) for q in existing_mcq}
    out = []
    for q in new_mcq or []:
        key = norm(q.get("prompt", ""))
        if key and key not in seen:
            seen.add(key)
            out.append(q)
    return out


# -----------------------------
# PDF text extraction by page range
# -----------------------------
def extract_pdf_text_pages(pdf_bytes: bytes, start_page: int, end_page: int):
    """
    User inputs are 1-based and inclusive.
    Returns (text, total_pages, actual_start, actual_end)
    """
    reader = PdfReader(BytesIO(pdf_bytes))
    total = len(reader.pages)
    if total == 0:
        return "", 0, 0, 0

    # Clamp into [1, total]
    start = max(1, min(start_page, total))
    end = max(1, min(end_page, total))
    if end < start:
        start, end = end, start

    chunks = []
    for i in range(start - 1, end):
        txt = reader.pages[i].extract_text() or ""
        txt = txt.strip()
        if txt:
            chunks.append(txt)

    return "\n\n".join(chunks), total, start, end


def generate_next_20_from_text(extracted_text: str, avoid_prompts):
    schema = mcq_schema()

    # Keep prompt size reasonable
    avoid_prompts = avoid_prompts[-80:]
    avoid_text = "\n".join(f"- {p}" for p in avoid_prompts) if avoid_prompts else "(none)"

    # Safety limit: if range is huge, extracted text might be too long
    # Truncate to reduce cost/timeouts; encourage smaller ranges if needed.
    MAX_CHARS = 60_000
    if len(extracted_text) > MAX_CHARS:
        extracted_text = extracted_text[:MAX_CHARS]

    instructions = f"""
You are generating revision questions from the provided study material text ONLY.

Generate EXACTLY 20 Multiple Choice Questions (MCQ).

Rules:
- Each question must test a DIFFERENT concept/fact from the text.
- Do NOT repeat or paraphrase any existing questions listed below.
- Exactly 4 options (A–D), only ONE correct.
- No explanations.

Existing questions to avoid:
{avoid_text}
"""

    resp = client.responses.create(
        model="gpt-5",
        instructions=instructions,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": extracted_text}
            ]
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

    parsed = json.loads(resp.output_text)
    return parsed.get("mcq", [])


# -----------------------------
# API endpoints
# -----------------------------
@app.route("/api/reset", methods=["POST"])
def reset():
    cleanup_sessions()
    session_id = request.form.get("session_id") or ""
    if session_id:
        SESSIONS.pop(session_id, None)
    return jsonify({"ok": True})


@app.route("/api/next-20", methods=["POST"])
def next_20():
    cleanup_sessions()

    session_id = request.form.get("session_id") or ""
    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400

    pdf = request.files.get("pdf")
    if not pdf:
        return jsonify({"error": "No PDF uploaded (field name must be 'pdf')"}), 400

    data = pdf.read()
    if not data:
        return jsonify({"error": "Empty PDF"}), 400

    # Read page range
    try:
        start_page = int(request.form.get("start_page", "1"))
        end_page = int(request.form.get("end_page", "1"))
    except ValueError:
        return jsonify({"error": "Start/End page must be numbers"}), 400

    pdf_hash = hash(data)

    # Initialize or validate session
    s = SESSIONS.get(session_id)
    if s is None:
        s = {"pdf_hash": pdf_hash, "mcq": [], "updated": time.time()}
        SESSIONS[session_id] = s
    else:
        if s.get("pdf_hash") != pdf_hash:
            return jsonify({"error": "PDF does not match the one you started with. Click Reset and start again."}), 400

    extracted_text, total_pages, actual_start, actual_end = extract_pdf_text_pages(
        data, start_page, end_page
    )

    if total_pages == 0:
        return jsonify({"error": "Could not read PDF pages."}), 400

    if not extracted_text.strip():
        return jsonify({
            "error": "No readable text found in that page range. If this PDF is scanned (image-only), text extraction won’t work.",
            "total_pages": total_pages
        }), 400

    existing_prompts = [q.get("prompt", "") for q in s["mcq"]]

    print(f"[{session_id}] Next 20 from pages {actual_start}-{actual_end} (total pages {total_pages}). Existing={len(s['mcq'])}", flush=True)
    new_mcq = generate_next_20_from_text(extracted_text, existing_prompts)

    new_unique = dedupe(s["mcq"], new_mcq)
    s["mcq"].extend(new_unique)
    s["updated"] = time.time()

    return jsonify({
        "mcq": s["mcq"],
        "added": len(new_unique),
        "total_pages": total_pages,
        "used_range": {"start": actual_start, "end": actual_end}
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
