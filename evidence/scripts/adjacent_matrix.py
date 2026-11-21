# Disposable fixture for issue 49929 baseline verification. Offline mock Anthropic; one nonce prompt per case.
import asyncio, json, threading, time, uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
import litellm
from litellm.caching.caching import Cache
from readiness import cache_entries_ready

litellm.suppress_debug_info = True
litellm.cache = Cache(type="local")

TOOL = {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}}
TEXT = {"type": "text", "text": "sunny"}
SHAPES = {"tool_use+end_turn": (TOOL, "end_turn"), "tool_use+tool_use": (TOOL, "tool_use"),
          "text+end_turn": (TEXT, "end_turn"), "text+max_tokens": (TEXT, "max_tokens")}
HITS = {}

def body(block, stop):
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-4-5", "content": [block],
            "stop_reason": stop, "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1}}

def events(block, stop):
    if block["type"] == "tool_use":
        start, delta = block, {"type": "input_json_delta", "partial_json": "{}"}
    else:
        start, delta = {"type": "text", "text": ""}, {"type": "text_delta", "text": block["text"]}
    return [("message_start", {"type": "message_start", "message": dict(body(block, stop), content=[], stop_reason=None)}),
            ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": start}),
            ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": delta}),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            ("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None},
                               "usage": {"output_tokens": 1}}),
            ("message_stop", {"type": "message_stop"})]

class Mock(BaseHTTPRequestHandler):
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = req["messages"][-1]["content"]
        if isinstance(prompt, list):
            prompt = prompt[0]["text"]
        shape, nonce = prompt.split("|")[:2]
        HITS[nonce] = HITS.get(nonce, 0) + 1
        block, stop = SHAPES[shape]
        if req.get("stream"):
            data = "".join("event: %s\ndata: %s\n\n" % (n, json.dumps(e)) for n, e in events(block, stop)).encode()
            ctype = "text/event-stream"
        else:
            data, ctype = json.dumps(body(block, stop)).encode(), "application/json"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def log_message(self, *a): pass

srv = HTTPServer(("127.0.0.1", 0), Mock)
threading.Thread(target=srv.serve_forever, daemon=True).start()
BASE = "http://127.0.0.1:%d" % srv.server_port

def args(shape, nonce):
    return dict(model="anthropic/claude-sonnet-4-5", max_tokens=100,
                messages=[{"role": "user", "content": "%s|%s|weather?" % (shape, nonce)}], api_base=BASE, api_key="x")

def summarize_chunks(chunks):
    last = [c.choices[0].finish_reason for c in chunks if c.choices and c.choices[0].finish_reason]
    tool = any(c.choices and c.choices[0].delta.tool_calls for c in chunks)
    built = litellm.stream_chunk_builder(chunks).choices[0]
    return {"last_chunk": last[-1] if last else None, "builder": built.finish_reason,
            "tool_calls": tool, "builder_tool_calls": bool(built.message.tool_calls), "id": chunks[0].id}

def plain(a):
    r = litellm.completion(**a)
    return {"finish_reason": r.choices[0].finish_reason, "tool_calls": bool(r.choices[0].message.tool_calls),
            "cache_hit": r._hidden_params.get("cache_hit"), "id": r.id}

def stream(a):
    return summarize_chunks(list(litellm.completion(**a, stream=True)))

async def aplain_(a):
    r = await litellm.acompletion(**a)
    return {"finish_reason": r.choices[0].finish_reason, "tool_calls": bool(r.choices[0].message.tool_calls),
            "cache_hit": r._hidden_params.get("cache_hit"), "id": r.id}

async def astream_(a):
    resp = await litellm.acompletion(**a, stream=True)
    return summarize_chunks([c async for c in resp])

LOOP = asyncio.new_event_loop()
def aplain(a): return LOOP.run_until_complete(aplain_(a))
def astream(a): return LOOP.run_until_complete(astream_(a))
MODES = {"plain": plain, "stream": stream, "aplain": aplain, "astream": astream}

def stored(before):
    store = litellm.cache.cache.cache_dict
    out = []
    for k in store:
        if k in before:
            continue
        v = store[k]
        if isinstance(v, dict) and "response" in v:
            v = v["response"]
        if isinstance(v, str):
            v = json.loads(v)
        try:
            out.append(v["choices"][0].get("finish_reason"))
        except Exception as e:
            out.append("unreadable:%s" % type(v).__name__)
    return out

def assert_hit(miss, hit, hits_after_miss, hits_total):
    assert hits_after_miss == 1 and hits_total == 1, "provider reached %s then %s times" % (hits_after_miss, hits_total)
    assert miss["id"] == hit["id"], "cache hit returned a different response id"
    if "cache_hit" in hit:
        assert hit["cache_hit"] is True, "cache_hit marker missing on the second call"

def case(shape, miss_mode, hit_mode):
    nonce = uuid.uuid4().hex[:8]
    a = args(shape, nonce)
    before = set(litellm.cache.cache.cache_dict)
    miss = MODES[miss_mode](a)
    cache_entries_ready(LOOP, before)
    st = stored(before)
    hits_after_miss = HITS.get(nonce, 0)
    hit = MODES[hit_mode](a)
    same_mode = miss_mode == hit_mode
    if same_mode:
        assert_hit(miss, hit, hits_after_miss, HITS.get(nonce, 0))
    row = {"shape": shape, "miss_mode": miss_mode, "hit_mode": hit_mode, "miss": miss, "stored_finish_reason": st,
           "hit": hit, "provider_hits_after_miss": hits_after_miss, "provider_hits_total": HITS.get(nonce, 0),
           "hit_asserted": same_mode}
    print(json.dumps(row), flush=True)

def main():
  print(json.dumps({"litellm_file": litellm.__file__}))
  for shape in SHAPES:
    for m in MODES:
        case(shape, m, m)
  for miss_mode, hit_mode in [("plain", "stream"), ("stream", "plain"), ("aplain", "astream"), ("astream", "aplain")]:
    case("tool_use+end_turn", miss_mode, hit_mode)

if __name__ == "__main__":
    main()
