# Anthropic answers with a tool_use block and stop_reason "end_turn". With the cache on, the
# same litellm.completion() call is made twice, non-streaming and with stream=True. The second
# call of each pair is a cache hit and should return the same finish_reason as the first.
import json, sys, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import litellm
from litellm.caching.caching import Cache

litellm.suppress_debug_info = True
litellm.cache = Cache(type="local")

BODY = {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-4-5",
        "content": [{"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}}],
        "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1}}
EVENTS = [
    ("message_start", {"type": "message_start", "message": dict(BODY, content=[], stop_reason=None)}),
    ("content_block_start", {"type": "content_block_start", "index": 0,
                             "content_block": {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}}}),
    ("content_block_delta", {"type": "content_block_delta", "index": 0,
                             "delta": {"type": "input_json_delta", "partial_json": "{}"}}),
    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                       "usage": {"output_tokens": 1}}),
    ("message_stop", {"type": "message_stop"}),
]
HITS = []

class MockAnthropic(BaseHTTPRequestHandler):
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        HITS.append(1)
        if req.get("stream"):
            data = "".join("event: %s\ndata: %s\n\n" % (n, json.dumps(e)) for n, e in EVENTS).encode()
            ctype = "text/event-stream"
        else:
            data, ctype = json.dumps(BODY).encode(), "application/json"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def log_message(self, *a): pass

srv = HTTPServer(("127.0.0.1", 0), MockAnthropic)
threading.Thread(target=srv.serve_forever, daemon=True).start()
args = dict(model="anthropic/claude-sonnet-4-5", max_tokens=100,
            messages=[{"role": "user", "content": "weather?"}],
            api_base="http://127.0.0.1:%d" % srv.server_port, api_key="x")

def plain():
    return litellm.completion(**args).choices[0].finish_reason

def streamed():
    chunks = list(litellm.completion(**args, stream=True))
    last = [c.choices[0].finish_reason for c in chunks if c.choices and c.choices[0].finish_reason][-1]
    return last, litellm.stream_chunk_builder(chunks).choices[0].finish_reason

print("stream=False call 1:", plain())
print("stream=False call 2:", plain())
print("stream=True  call 1 (last chunk, stream_chunk_builder):", streamed())
print("stream=True  call 2 (last chunk, stream_chunk_builder):", streamed())
print("requests that reached the mock:", len(HITS))
