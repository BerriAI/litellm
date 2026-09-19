"""
X-Trust Hook for LiteLLM Proxy
Annotates requests with human presence score via X-Trust header.
Zero KYC, zero PII. github.com/htl-syterme/htl-core
"""
import hmac
import hashlib
import base64
import json
import time
from typing import Optional
from litellm.integrations.custom_logger import CustomLogger


class XTrustHook(CustomLogger):
    """
    LiteLLM hook that validates X-Trust headers.
    Annotates requests with human presence score.
    Doctrine AIR: annotate, never block.
    """

    def __init__(self, secret: str, min_score: float = 0.0):
        self.secret = secret
        self.min_score = min_score

    def _verify(self, token: str) -> Optional[dict]:
        try:
            parts = token.split(".")
            if len(parts) != 3 or parts[0] != "v1":
                return None
            _, payload_b64, sig_b64 = parts
            padding = 4 - len(payload_b64) % 4
            payload_bytes = base64.b64decode(
                payload_b64.replace("-", "+").replace("_", "/") + "=" * padding
            )
            sig_bytes = base64.b64decode(
                sig_b64.replace("-", "+").replace("_", "/") + "=" * (4 - len(sig_b64) % 4)
            )
            expected = hmac.new(
                self.secret.encode(), payload_b64.encode(), hashlib.sha256
            ).digest()
            if not hmac.compare_digest(expected, sig_bytes):
                return None
            payload = json.loads(payload_bytes)
            now = int(time.time())
            if now > payload.get("exp", 0):
                return None
            if now - payload.get("iat", 0) > 120:
                return None
            score = payload.get("score", 0)
            if not (0 <= score <= 1):
                return None
            return payload
        except Exception:
            return None

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        token = data.get("metadata", {}).get("headers", {}).get("x-trust", "")
        payload = self._verify(token) if token else None
        score = payload["score"] if payload else 0.0
        trusted = payload is not None and score >= self.min_score
        # Redact token after verification to prevent replay via logging
        if "headers" in data.get("metadata", {}):
            data["metadata"]["headers"].pop("x-trust", None)
        if "metadata" not in data:
            data["metadata"] = {}
        data["metadata"]["x_trust"] = {
            "trusted": trusted,
            "score": score,
            "annotated": True,
        }
        return data
