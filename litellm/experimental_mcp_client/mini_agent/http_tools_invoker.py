from __future__ import annotations
import json
from typing import Any, Dict, List, Optional

import types as _types
try:
    import httpx  # type: ignore
except Exception:
    httpx = _types.SimpleNamespace(AsyncClient=object)  # type: ignore


class HttpToolsInvoker:
    def __init__(self, base_url: str, headers: Optional[Dict[str, str]] = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = headers or {}

    async def list_openai_tools(self) -> List[Dict[str, Any]]:
        async with httpx.AsyncClient(headers=self.headers) as client:  # type: ignore
            r = await client.get(f"{self.base_url}/tools")
            if getattr(r, "status_code", 200) >= 400:
                tail = (getattr(r, "text", "") or "")[:256]
                raise Exception(f"HTTP {getattr(r,'status_code',0)}: {tail}")
            return r.json()

    async def call_openai_tool(self, openai_tool: Dict[str, Any]) -> str:
        fn = (openai_tool or {}).get("function", {})
        name = fn.get("name", "")
        args = fn.get("arguments", {})
        body = {"name": name, "arguments": args}
        async with httpx.AsyncClient(headers=self.headers) as client:  # type: ignore
            r = await client.post(f"{self.base_url}/invoke", json=body)
            if getattr(r, "status_code", 200) >= 400:
                tail = (getattr(r, "text", "") or "")[:256]
                raise Exception(f"HTTP {getattr(r,'status_code',0)}: {tail}")
            data = r.json()
            if isinstance(data, dict):
                if "text" in data:
                    return data["text"]
                if "error" in data:
                    raise Exception(data["error"])  # pragma: no cover
            return json.dumps(data)
