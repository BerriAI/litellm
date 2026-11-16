"""Disposable upstream for the PR2 E2E. The mode is read from the LAST user message, outside the cached prefix.
MODE:real forwards to api.anthropic.com unchanged. Every other mode is a labelled FIXTURE reply:
write (write=2048), hit (read=2048), miss (0/0), nousage (no cache counters), fail (HTTP 500). stream=true gets a canned 0/0 SSE stream."""
import json, time, httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
app = FastAPI(); LOG = []
def usage(mode):
    u = {"input_tokens": 20, "output_tokens": 1}
    if mode != "nousage": u.update(cache_creation_input_tokens=2048 if mode == "write" else 0, cache_read_input_tokens=2048 if mode == "hit" else 0)
    return u
def canned(mode):
    return {"id": f"msg_fx{int(time.time()*1e6)}", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5",
            "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "stop_sequence": None, "usage": usage(mode)}
def sse(t, p): return f"event: {t}\ndata: {json.dumps({'type': t, **p})}\n\n".encode()
async def canned_stream(mode):
    m = canned(mode); m["content"] = []; m["stop_reason"] = None
    for f in (sse("message_start", {"message": m}), sse("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}),
              sse("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "ok"}}), sse("content_block_stop", {"index": 0}),
              sse("message_delta", {"delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 1}}), sse("message_stop", {})):
        yield f
def mode_of(messages):
    last = messages[-1]["content"]; text = last if isinstance(last, str) else " ".join(b.get("text", "") for b in last if isinstance(b, dict))
    return text.split("MODE:")[1].split()[0] if "MODE:" in text else "miss"
@app.post("/{tag}/v1/messages")
async def messages(tag: str, r: Request):
    raw = await r.body(); body = json.loads(raw); stream = bool(body.get("stream")); mode = mode_of(body["messages"])
    rec = {"tag": tag, "mode": mode, "stream": stream, "t": time.time()}
    if mode == "real":
        hdrs = {k: v for k, v in r.headers.items() if k.lower() in ("x-api-key", "anthropic-version", "anthropic-beta", "content-type")}
        async with httpx.AsyncClient(timeout=120) as c:
            up = await c.post("https://api.anthropic.com/v1/messages", content=raw, headers=hdrs)
        LOG.append({**rec, "status": up.status_code, "source": "anthropic"})
        return Response(content=up.content, status_code=up.status_code, media_type=up.headers.get("content-type", "application/json"))
    if mode == "fail":
        LOG.append({**rec, "status": 500, "source": "fixture"}); return JSONResponse({"type": "error", "error": {"type": "api_error", "message": "injected"}}, status_code=500)
    LOG.append({**rec, "status": 200, "source": "fixture"})
    return StreamingResponse(canned_stream(mode), media_type="text/event-stream") if stream else JSONResponse(canned(mode))
@app.post("/oai/v1/chat/completions")
async def oai(r: Request):
    body = await r.json(); LOG.append({"tag": "oai", "mode": mode_of(body["messages"]), "t": time.time(), "status": 200, "source": "fixture"})
    return JSONResponse({"id": "chatcmpl-fx", "object": "chat.completion", "created": int(time.time()), "model": "fixture", "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": 20, "completion_tokens": 1, "total_tokens": 21, "prompt_tokens_details": {"cached_tokens": 0}}})
@app.get("/_log")
async def get_log(): return LOG
