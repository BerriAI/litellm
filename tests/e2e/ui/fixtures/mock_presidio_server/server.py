"""
Mock Presidio analyzer + anonymizer for UI e2e tests.
Serves POST /analyze and POST /anonymize over a fixed set of regex recognizers.
"""

import os
import re

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware


def _luhn_ok(candidate):
    digits = [int(c) for c in candidate if c.isdigit()]
    doubled = [d * 2 - 9 if d * 2 > 9 else d * 2 for d in digits[-2::-2]]
    return len(digits) >= 13 and (sum(digits[::-1][::2]) + sum(doubled)) % 10 == 0


RECOGNIZERS = (
    ("EMAIL_ADDRESS", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), 1.0, None),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b"), 1.0, _luhn_ok),
    ("US_SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), 0.85, None),
    ("PHONE_NUMBER", re.compile(r"\b(?:\+?\d{1,2}[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]?\d{3}[ .-]?\d{4}\b"), 0.75, None),
    ("IP_ADDRESS", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), 0.6, None),
    ("URL", re.compile(r"\bhttps?://[^\s]+"), 0.5, None),
)

app = FastAPI(title="Mock Presidio Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _detect(text, wanted):
    found = [
        {"entity_type": name, "start": m.start(), "end": m.end(), "score": score}
        for name, pattern, score, validate in RECOGNIZERS
        if wanted is None or name in wanted
        for m in pattern.finditer(text)
        if validate is None or validate(m.group())
    ]
    ranked = sorted(found, key=lambda r: (-r["score"], r["start"]))
    kept = []
    for candidate in ranked:
        overlaps = any(candidate["start"] < k["end"] and k["start"] < candidate["end"] for k in kept)
        if not overlaps:
            kept.append(candidate)
    return sorted(kept, key=lambda r: r["start"])


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(request: Request):
    body = await request.json()
    text = body.get("text", "") or ""
    entities = body.get("entities")
    wanted = set(entities) if entities else None
    threshold = body.get("score_threshold") or 0.0
    return [
        {**result, "analysis_explanation": None, "recognition_metadata": {"recognizer_name": "MockRecognizer"}}
        for result in _detect(text, wanted)
        if result["score"] >= threshold
    ]


@app.post("/anonymize")
async def anonymize(request: Request):
    body = await request.json()
    text = body.get("text", "") or ""
    results = sorted(body.get("analyzer_results") or [], key=lambda r: r["start"])

    pieces = []
    items = []
    cursor = 0
    for result in results:
        start, end = result["start"], result["end"]
        if start < cursor:
            continue
        placeholder = f"<{result['entity_type']}>"
        pieces.append(text[cursor:start])
        masked_start = sum(len(p) for p in pieces)
        pieces.append(placeholder)
        items.append(
            {
                "operator": "replace",
                "entity_type": result["entity_type"],
                "start": masked_start,
                "end": masked_start + len(placeholder),
                "text": placeholder,
            }
        )
        cursor = end
    pieces.append(text[cursor:])

    return {"text": "".join(pieces), "items": list(reversed(items))}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("MOCK_PRESIDIO_PORT", "8091")))
