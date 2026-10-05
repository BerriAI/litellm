import json
from hashlib import sha256
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server

_GEMINI_MODEL: Final = "gemini/gemini-2.5-flash"
_GEMINI_DEPLOYMENT_KEY: Final = "synthetic-gemini-deployment-key"
_CONTENTS: Final = ({"role": "user", "parts": [{"text": "spend probe"}]},)
_USAGE: Final = {"promptTokenCount": 11, "candidatesTokenCount": 7, "totalTokenCount": 18}

_REPLY_BODY: Final = {
    "candidates": [
        {"content": {"parts": [{"text": "ok"}], "role": "model"}, "finishReason": "STOP", "index": 0}
    ],
    "usageMetadata": _USAGE,
}

_STREAM_FRAMES: Final = (
    b'data: {"candidates":[{"content":{"parts":[{"text":"ok"}],"role":"model"},"index":0}]}\n\n',
    (
        b'data: {"candidates":[{"content":{"parts":[],"role":"model"},"finishReason":"STOP","index":0}],'
        b'"usageMetadata":{"promptTokenCount":11,"candidatesTokenCount":7,"totalTokenCount":18}}\n\n'
    ),
)

_CALL_TYPE: Final = {False: "agenerate_content", True: "agenerate_content_stream"}


@pytest.mark.parametrize("stream", [False, True])
def test_generate_content_bills_from_usage_metadata(gateway: Gateway, stream: bool) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.headers["x-goog-api-key"] == _GEMINI_DEPLOYMENT_KEY
        if stream:
            assert request.target == "/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
            return Reply(content_type="text/event-stream", chunks=_STREAM_FRAMES)
        assert request.target == "/models/gemini-2.5-flash:generateContent"
        return Reply(body=json.dumps(_REPLY_BODY).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=_GEMINI_MODEL,
            api_base=wire.url,
            api_key=_GEMINI_DEPLOYMENT_KEY,
            input_cost_per_token=0.001,
            output_cost_per_token=0.002,
        )
        key: Final = scenario.key(models=[model], max_budget=10)
        query: Final = "?alt=sse" if stream else ""
        action: Final = "streamGenerateContent" if stream else "generateContent"
        response: Final = gateway.request(
            "POST",
            f"/v1beta/models/{model}:{action}{query}",
            {"contents": list(_CONTENTS)},
            key=key,
        )
        assert response.status_code == 200, response.text
        if not stream:
            assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(0.025), response.headers
        digest: Final = sha256(key.encode()).hexdigest()
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT call_type, prompt_tokens, completion_tokens, spend FROM "LiteLLM_SpendLogs" WHERE api_key=%s',
                (digest,),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        assert rows[0]["call_type"] == _CALL_TYPE[stream], rows
        assert rows[0]["prompt_tokens"] == 11, rows
        assert rows[0]["completion_tokens"] == 7, rows
        assert float(rows[0]["spend"]) == pytest.approx(0.025), rows
        info: Final = eventually(
            lambda: object_value(gateway.get("/key/info", {"key": digest})["info"]),
            lambda values: float(values.get("spend", 0)) > 0,
            seconds=70,
        )
        assert float(info["spend"]) == pytest.approx(0.025), info
        reread: Final = read_rows(
            'SELECT call_type, prompt_tokens, completion_tokens, spend FROM "LiteLLM_SpendLogs" WHERE api_key=%s',
            (digest,),
        )
        assert reread == rows, reread
        assert [(r.method, r.target) for r in wire.drain()] == [
            ("POST", f"/models/gemini-2.5-flash:{action}{'?alt=sse' if stream else ''}")
        ]
