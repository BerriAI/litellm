# Offline mock Anthropic upstream for the proxy legs of issue 49929. Prompt is "<shape>|<nonce>|...".
# GET /hits returns how many requests reached the mock per nonce.
import json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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

def prompt_of(req):
    content = req["messages"][-1]["content"]
    return content if isinstance(content, str) else content[0]["text"]

class Mock(BaseHTTPRequestHandler):
    def send(self, data, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def do_GET(self):
        self.send(json.dumps(HITS).encode(), "application/json")
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        shape, nonce = prompt_of(req).split("|")[:2]
        HITS[nonce] = HITS.get(nonce, 0) + 1
        block, stop = SHAPES[shape]
        if req.get("stream"):
            self.send("".join("event: %s\ndata: %s\n\n" % (n, json.dumps(e)) for n, e in events(block, stop)).encode(), "text/event-stream")
        else:
            self.send(json.dumps(body(block, stop)).encode(), "application/json")
    def log_message(self, *a): pass

ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Mock).serve_forever()
