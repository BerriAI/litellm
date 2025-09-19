from __future__ import annotations
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import http_tools_invoker as inv
from . import litellm_mcp_mini_agent as agent


class AgentRunReq(BaseModel):
    messages: List[Dict[str, Any]]
    model: str
    tool_backend: str = "http"
    tool_http_base_url: Optional[str] = None
    tool_http_headers: Optional[Dict[str, str]] = None


app = FastAPI()


@app.post("/agent/run")
async def run(req: AgentRunReq):
    if req.tool_backend == "http":
        if not req.tool_http_base_url:
            raise HTTPException(status_code=400, detail="tool_http_base_url required for tool_backend=http")
        mcp = inv.HttpToolsInvoker(req.tool_http_base_url, headers=req.tool_http_headers)
    else:
        mcp = None

    if not hasattr(agent, "arouter_call"):
        raise HTTPException(status_code=500, detail="agent.arouter_call missing")

    result = await agent.arouter_call(model=req.model, messages=req.messages, stream=False)
    return result
