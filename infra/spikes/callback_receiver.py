"""
Flask callback receiver for the EC2 provisioning spike (LIT-2888).

Listens on port 3333 for POST /spike. Logs every callback with timestamp + body.
Used in tandem with ngrok to give EC2 instances a public HTTPS URL to hit.

Run:
    uv run python infra/spikes/callback_receiver.py
"""

import json
import sys
import time
from flask import Flask, request

app = Flask(__name__)

# In-memory log of received callbacks. Useful when the spike script polls
# the receiver to confirm both `phase=boot` and `phase=hydrate` arrived.
CALLBACKS: list = []


@app.post("/spike")
def spike() -> tuple[dict, int]:
    body = request.get_json(silent=True) or {}
    entry = {"received_at": time.time(), "body": body}
    CALLBACKS.append(entry)
    print(
        f"[{time.strftime('%H:%M:%S')}] callback: {json.dumps(body)}",
        file=sys.stderr,
        flush=True,
    )
    return {"ok": True}, 200


@app.get("/callbacks")
def list_callbacks() -> dict:
    """Return all callbacks received so far. The spike script polls this."""
    return {"callbacks": CALLBACKS}


@app.get("/health")
def health() -> dict:
    return {"ok": True}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3333, debug=False)
