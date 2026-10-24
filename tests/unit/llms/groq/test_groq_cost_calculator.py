import pytest

from litellm.llms.groq.cost_calculator import cost_per_web_search_request
from litellm.types.utils import ModelInfo, ServerToolUse, Usage

PRICED_MODEL_INFO = ModelInfo(
    key="groq/openai/gpt-oss-20b",
    litellm_provider="groq",
    mode="chat",
    search_context_cost_per_query={
        "search_context_size_low": 0.005,
        "search_context_size_medium": 0.005,
        "search_context_size_high": 0.005,
    },
)


def _usage_with_actions(searches: int | None, opens: int | None = None) -> Usage:
    return Usage(
        prompt_tokens=100,
        completion_tokens=10,
        total_tokens=110,
        server_tool_use=ServerToolUse(web_search_requests=searches, browser_open_requests=opens),
    )


def test_bills_per_executed_search():
    cost = cost_per_web_search_request(usage=_usage_with_actions(7), model_info=PRICED_MODEL_INFO)
    assert cost == pytest.approx(7 * 0.005)


def test_no_searches_costs_nothing():
    cost = cost_per_web_search_request(usage=_usage_with_actions(None), model_info=PRICED_MODEL_INFO)
    assert cost == 0.0


def test_missing_usage_signal_costs_nothing():
    usage = Usage(prompt_tokens=100, completion_tokens=10, total_tokens=110)
    cost = cost_per_web_search_request(usage=usage, model_info=PRICED_MODEL_INFO)
    assert cost == 0.0


def test_missing_pricing_costs_nothing():
    unpriced = ModelInfo(key="groq/openai/gpt-oss-20b", litellm_provider="groq", mode="chat")
    cost = cost_per_web_search_request(usage=_usage_with_actions(3), model_info=unpriced)
    assert cost == 0.0


def test_bills_visit_website_per_open():
    cost = cost_per_web_search_request(usage=_usage_with_actions(0, opens=15), model_info=PRICED_MODEL_INFO)
    assert cost == pytest.approx(15 * 0.001)


def test_bills_searches_and_opens_together():
    cost = cost_per_web_search_request(usage=_usage_with_actions(2, opens=3), model_info=PRICED_MODEL_INFO)
    assert cost == pytest.approx(2 * 0.005 + 3 * 0.001)


def test_groq_openai_aliased_completion_cost():
    """Regression test for Issue #48226: completion_cost for Groq models prefixed with openai/."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    import litellm
    from litellm.types.utils import Choices, Message, ModelResponse

    body = {
        "id": "chatcmpl-x",
        "object": "chat.completion",
        "created": 1761100000,
        "model": "openai/gpt-oss-120b",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 77, "completion_tokens": 19, "total_tokens": 96},
        "service_tier": "on_demand",
        "system_fingerprint": "fp_x",
        "x_groq": {"id": "req_x"},
    }

    class MockHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("content-length", 0)))
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), MockHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    resp = litellm.completion(
        model="groq/openai/gpt-oss-120b",
        messages=[{"role": "user", "content": "hi"}],
        api_base=f"http://127.0.0.1:{srv.server_port}/openai/v1",
        api_key="test",
    )

    assert resp._hidden_params.get("custom_llm_provider") == "groq"
    assert resp.model == "groq/openai/gpt-oss-120b"
    assert resp._hidden_params.get("response_cost") == pytest.approx(2.295e-05)

    c1 = litellm.completion_cost(completion_response=resp)
    assert c1 == pytest.approx(2.295e-05)

    c2 = litellm.completion_cost(completion_response=resp, model="groq/openai/gpt-oss-120b")
    assert c2 == pytest.approx(2.295e-05)

    c3 = litellm.completion_cost(completion_response=resp, custom_llm_provider="groq")
    assert c3 == pytest.approx(2.295e-05)

    # Raw response without hidden params must also resolve when explicit model is passed
    raw_resp = ModelResponse(
        id="chatcmpl-x",
        model="openai/gpt-oss-120b",
        choices=[Choices(message=Message(content="ok", role="assistant"))],
        usage=Usage(prompt_tokens=77, completion_tokens=19, total_tokens=96),
    )
    c_raw = litellm.completion_cost(completion_response=raw_resp, model="groq/openai/gpt-oss-120b")
    assert c_raw == pytest.approx(2.295e-05)

