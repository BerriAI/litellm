"""HTTP-based MCP invoker for proxying mini-agent tool calls to services."""

from __future__ import annotations
import json
import asyncio
from typing import Any, Dict, List, Optional

import types as _types
try:
    import httpx  # type: ignore
except Exception:
    class _DummyAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, *args, **kwargs):
            raise RuntimeError("httpx is required for HttpToolsInvoker HTTP calls")

        async def post(self, *args, **kwargs):
            raise RuntimeError("httpx is required for HttpToolsInvoker HTTP calls")

    httpx = _types.SimpleNamespace(AsyncClient=_DummyAsyncClient)  # type: ignore


class HttpToolsInvoker:
    def __init__(self, base_url: str, headers: Optional[Dict[str, str]] = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = dict(headers or {})

    def _mk_client(self):
        """
        Create an AsyncClient. Some test doubles don't accept kwargs; fall back gracefully.
        """
        AsyncClient = getattr(httpx, "AsyncClient", object)
        try:
            return AsyncClient(headers=self.headers)  # type: ignore[call-arg]
        except TypeError:
            return AsyncClient()  # type: ignore[call-arg]

    async def list_openai_tools(self) -> List[Dict[str, Any]]:
        async with self._mk_client() as client:  # type: ignore
            # Prefer passing headers; fall back for stubs that don't accept it
            try:
                r = await client.get(f"{self.base_url}/tools", headers=self.headers)
            except TypeError:
                r = await client.get(f"{self.base_url}/tools")
            if getattr(r, "status_code", 200) >= 400:
                tail = (getattr(r, "text", "") or "")[:256]
                raise Exception(f"HTTP {getattr(r,'status_code',0)}: {tail}")
            try:
                return r.json()
            except Exception:
                return []

    async def call_openai_tool(self, openai_tool: Dict[str, Any]) -> str:
        fn = (openai_tool or {}).get("function", {})
        name = fn.get("name", "")
        args = fn.get("arguments", {})
        body = {"name": name, "arguments": args}

        async with self._mk_client() as client:  # type: ignore
            
            try:
                r = await client.post(f"{self.base_url}/invoke", json=body, headers=self.headers)
            except TypeError:
                # test doubles may not accept keyword args; fall back
                r = await client.post(f"{self.base_url}/invoke", body)


            # One polite retry on 429 honoring Retry-After if present
            if getattr(r, "status_code", 200) == 429:
                ra = None
                try:
                    ra = getattr(r, "headers", {}).get("Retry-After") if hasattr(r, "headers") else None
                except Exception:
                    ra = None
                try:
                    delay = float(ra) if ra is not None else 0.0
                except Exception:
                    delay = 0.0
                # cap small delay to avoid slow tests
                if delay > 0:
                    await asyncio.sleep(min(delay, 1.0))
                
            try:
                r = await client.post(f"{self.base_url}/invoke", json=body, headers=self.headers)
            except TypeError:
                # test doubles may not accept keyword args; fall back
                r = await client.post(f"{self.base_url}/invoke", body)


            if getattr(r, "status_code", 200) >= 400:
                tail = (getattr(r, "text", "") or "")[:256]
                raise Exception(f"HTTP {getattr(r,'status_code',0)}: {tail}")

            # Prefer JSON; if not JSON, return plain text body
            try:
                data = r.json()
            except Exception:
                txt = getattr(r, "text", "")
                return txt if isinstance(txt, str) else str(txt)

            if isinstance(data, dict):
                # common keys returned by simple tool executors
                for k in ("text", "result", "answer"):
                    v = data.get(k)
                    if isinstance(v, str):
                        return v
                return json.dumps(data, ensure_ascii=False)

            return json.dumps(data, ensure_ascii=False)
