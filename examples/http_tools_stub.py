"""
Stub HTTP tools server for Mini-Agent HTTP tools adapter.

Endpoints:
  GET  /tools   -> list of OpenAI tool specs
  POST /invoke  -> execute named tool with JSON or 'arguments' JSON string

Run:
  uvicorn examples.http_tools_stub:app --host 127.0.0.1 --port 8791
"""
from __future__ import annotations

import json
from typing import Any, Dict, List
import asyncio

from fastapi import FastAPI, Request
import os
from fastapi.responses import JSONResponse


app = FastAPI()


@app.get("/tools")
async def list_tools() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Echo back the provided text",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "sleep",
                "description": "Sleep for ms milliseconds and return 'slept'",
                "parameters": {
                    "type": "object",
                    "properties": {"ms": {"type": "integer", "minimum": 0}},
                    "required": ["ms"],
                },
            },
        },
    ]


@app.post("/invoke")
async def invoke(req: Request) -> JSONResponse:
    try:
        gdelay = int(os.getenv("STUB_GLOBAL_DELAY_MS", "0"))
        if gdelay > 0:
            await asyncio.sleep(gdelay / 1000.0)
    except Exception:
        pass
    body = await req.json()
    name = body.get("name")
    args = body.get("arguments", {})
    if not isinstance(args, str):
        try:
            args = json.dumps(args)
        except Exception:
            args = "{}"
    if name == "echo":
        try:
            text = json.loads(args).get("text", "")
        except Exception:
            text = ""
        return JSONResponse({"ok": True, "text": text})
    if name == "sleep":
        try:
            ms = int(json.loads(args).get("ms", 0))
        except Exception:
            ms = 0
        await asyncio.sleep(max(0, ms) / 1000.0)
        return JSONResponse({"ok": True, "text": "slept"})
    return JSONResponse({"ok": False, "error": "tool_not_found"}, status_code=404)
