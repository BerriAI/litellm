from __future__ import annotations

import os
from typing import Any, Dict

import httpx


class CodeWorldProvider:
    def __init__(self, base: str, token: str | None = None):
        self.base = base.rstrip("/")
        self.token = token

    def _headers(self) -> Dict[str, str]:
        h: Dict[str, str] = {"Content-Type": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    async def acomplete(self, *, messages, metrics, iterations, allowed_languages, request_timeout: float, temperature: float | None = None, seed: int | None = None) -> Dict[str, Any]:
        payload = {
            "messages": messages,
            "codeworld_metrics": metrics,
            "codeworld_iterations": iterations,
            "codeworld_allowed_languages": allowed_languages,
            "request_timeout": float(request_timeout),
        }
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if seed is not None:
            payload["seed"] = int(seed)
        async with httpx.AsyncClient(timeout=float(request_timeout)+30.0) as client:
            r = await client.post(self.base + "/bridge/complete", json=payload, headers=self._headers())
            if r.status_code in (200, 202):
                return r.json()
            try:
                return {"error": True, "status": r.status_code, "body": r.json()}
            except Exception:
                return {"error": True, "status": r.status_code, "body": r.text}

