"""Mock research tools used by deterministic mini-agent demos and tests."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

try:
    import httpx  # optional; tests monkeypatch this
except Exception:  # pragma: no cover - optional dep
    httpx = None  # type: ignore


class ResearchPythonInvoker:
    """Lightweight HTTP research tools.

    - research_perplexity: POST {PPLX_API_BASE or https://api.perplexity.ai}/responses
      Returns {ok, answer, citations}
    - research_context7_docs: GET {C7_API_BASE}/search?library=...&topic=...
      Returns {ok, snippets}
    """

    async def list_openai_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "research_perplexity",
                    "description": "Query Perplexity for an answer with citations.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}, "max_tokens": {"type": "integer"}},
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "research_context7_docs",
                    "description": "Search Context7 docs and return snippets.",
                    "parameters": {
                        "type": "object",
                        "properties": {"library": {"type": "string"}, "topic": {"type": "string"}},
                        "required": ["library"],
                    },
                },
            },
        ]

    async def call_openai_tool(self, openai_tool: Dict[str, Any]) -> str:
        f = openai_tool.get("function", {})
        name = f.get("name")
        args_raw = f.get("arguments")
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
        except Exception:
            args = {}

        if name == "research_perplexity":
            if httpx is None:
                return json.dumps({"ok": False, "error": "httpx required"})
            key = os.getenv("PPLX_API_KEY")
            base = os.getenv("PPLX_API_BASE", "https://api.perplexity.ai")
            model = os.getenv("PPLX_MODEL", "pplx-70b-online")
            if not key:
                return json.dumps({"ok": False, "error": "PPLX_API_KEY not set"})
            payload: Dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": str(args.get("query", ""))}],
                "return_citations": True,
            }
            if args.get("max_tokens"):
                payload["max_tokens"] = int(args["max_tokens"])  # type: ignore
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:  # type: ignore
                    r = await client.post(
                        f"{base.rstrip('/')}/responses",
                        headers={"Authorization": f"Bearer {key}"},
                        json=payload,
                    )
                    r.raise_for_status()
                    data = r.json()
                    answer = None
                    citations: List[Dict[str, Any]] = []
                    try:
                        if isinstance(data.get("output"), list):
                            answer = data["output"][0].get("content")
                        answer = answer or data.get("text") or ""
                        citations = data.get("citations") or []
                    except Exception:
                        pass
                    return json.dumps({"ok": True, "answer": answer or "", "citations": citations}, ensure_ascii=False)
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)[:400]})

        if name == "research_context7_docs":
            if httpx is None:
                return json.dumps({"ok": False, "error": "httpx required"})
            base = os.getenv("C7_API_BASE")
            if not base:
                return json.dumps({"ok": False, "error": "C7_API_BASE not set"})
            params = {"library": args.get("library"), "topic": args.get("topic")}
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:  # type: ignore
                    r = await client.get(f"{base.rstrip('/')}/search", params=params)
                    r.raise_for_status()
                    data = r.json()
                    snippets = data.get("snippets") or []
                    return json.dumps({"ok": True, "snippets": snippets}, ensure_ascii=False)
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)[:400]})

        return json.dumps({"ok": False, "error": f"unknown tool: {name}"})
