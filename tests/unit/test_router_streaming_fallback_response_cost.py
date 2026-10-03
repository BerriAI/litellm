import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import MagicMock

import pytest

import litellm
from litellm.router import Router


def test_apply_fallback_hidden_params_keeps_chunk_response_cost():
    chunk = litellm.ModelResponseStream(
        id="test",
        model="openai/internal-fallback",
        choices=[],
    )
    chunk._hidden_params = {"response_cost": None, "model_id": "chunk-model-id"}
    fallback_response = MagicMock()
    fallback_response._hidden_params = {
        "response_cost": 0.0,
        "api_base": "https://fallback.example",
    }

    Router._apply_fallback_hidden_params_to_item(
        fallback_item=chunk,
        prepared_fallback_hidden_params=Router._prepare_fallback_hidden_params(fallback_response),
    )

    assert chunk._hidden_params["response_cost"] is None
    assert chunk._hidden_params["api_base"] == "https://fallback.example"
    assert chunk._hidden_params["model_id"] == "chunk-model-id"


class _FakeOpenAI(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        group = self.path.strip("/").split("/")[0]
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        if group == "primary-model":
            error = {"error": {"message": "overloaded", "type": "server_error", "code": 500}}
            self.wfile.write(f"data: {json.dumps(error)}\n\n".encode())
        else:
            base = {"id": "c", "object": "chat.completion.chunk", "created": 1, "model": body["model"]}
            chunks = [
                {
                    **base,
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hi"}, "finish_reason": None}],
                },
                {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                {**base, "choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 12, "total_tokens": 24}},
            ]
            for chunk in chunks:
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


@pytest.mark.asyncio
async def test_streaming_fallback_chunks_carry_none_response_cost():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOpenAI)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    root = f"http://127.0.0.1:{server.server_address[1]}"

    router = litellm.Router(
        model_list=[
            {
                "model_name": "primary-model",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "sk-fake",
                    "api_base": f"{root}/primary-model/v1",
                },
            },
            {
                "model_name": "fallback-model",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "api_key": "sk-fake",
                    "api_base": f"{root}/fallback-model/v1",
                },
            },
        ],
        fallbacks=[{"primary-model": ["fallback-model"]}],
        num_retries=0,
    )

    stream = await router.acompletion(
        model="primary-model",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        stream_options={"include_usage": True},
    )
    chunk_costs = []
    usage_costs = []
    async for chunk in stream:
        chunk_costs.append(chunk._hidden_params.get("response_cost"))
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            usage_costs.append(getattr(usage, "cost", None))
    server.shutdown()

    assert len(chunk_costs) > 0
    assert chunk_costs == [None] * len(chunk_costs)
    assert len(usage_costs) > 0
    assert usage_costs[-1] is not None and usage_costs[-1] > 0
