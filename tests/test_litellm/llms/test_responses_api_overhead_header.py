"""LIT-1756 regression tests.

The bug: ``BaseLLMHTTPHandler.{async_,}response_api_handler`` called
``(a)sync_httpx_client.post(...)`` at four call sites without forwarding
``logging_obj=``.  The ``@track_llm_api_timing()`` decorator on
``AsyncHTTPHandler.post`` (and now on ``HTTPHandler.post``) reads
``logging_obj`` from kwargs to record ``llm_api_duration_ms``.  Without it
``response_metadata.set_timing_metrics`` leaves
``_hidden_params["litellm_overhead_time_ms"]`` as ``None`` and the proxy's
``get_custom_headers`` omits ``x-litellm-overhead-duration-ms`` from
``/v1/responses`` responses (chat completions are fine because that path
already forwards ``logging_obj``).
"""
import asyncio
import ast
import pathlib
import time
from unittest.mock import patch

import httpx
import pytest

from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
)
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler


def _fake_responses_payload() -> dict:
    return {
        "id": "resp_test_001",
        "object": "response",
        "created_at": int(time.time()),
        "model": "gpt-4o",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": "m1",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {"type": "output_text", "text": "Hello", "annotations": []}
                ],
            }
        ],
        "usage": {"input_tokens": 5, "output_tokens": 5, "total_tokens": 10},
    }


# ---- Decorator presence ----------------------------------------------------

def test_async_post_has_track_llm_api_timing_decorator() -> None:
    """``AsyncHTTPHandler.post`` must carry ``@track_llm_api_timing()``."""
    assert hasattr(AsyncHTTPHandler.post, "__wrapped__"), (
        "AsyncHTTPHandler.post is not decorated — track_llm_api_timing missing"
    )


def test_sync_post_has_track_llm_api_timing_decorator() -> None:
    """``HTTPHandler.post`` (sync) must also carry ``@track_llm_api_timing()``.

    Before LIT-1756 only the async post was decorated, so the sync code path
    in ``response_api_handler`` could never populate ``llm_api_duration_ms``.
    """
    assert hasattr(HTTPHandler.post, "__wrapped__"), (
        "HTTPHandler.post is not decorated — track_llm_api_timing missing"
    )


# ---- Call-site forwarding (source-level) -----------------------------------

@pytest.mark.parametrize(
    "method_name",
    ["response_api_handler", "async_response_api_handler"],
    ids=["sync", "async"],
)
def test_responses_api_handler_passes_logging_obj_to_post(method_name: str) -> None:
    """Every ``(a)sync_httpx_client.post(...)`` call inside the responses API
    handlers must include ``logging_obj=logging_obj`` as a kwarg.

    Source-level assertion via ``ast.parse`` so we don't depend on whether a
    given runtime test path exercises the streaming / fake-stream branches.
    """
    src_path = pathlib.Path(BaseLLMHTTPHandler.__module__.replace(".", "/") + ".py")
    if not src_path.exists():
        import litellm.llms.custom_httpx.llm_http_handler as _m
        src_path = pathlib.Path(_m.__file__)
    module_src = src_path.read_text()
    tree = ast.parse(module_src)
    target = None
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == method_name
        ):
            target = node
            break
    assert target is not None, f"could not find {method_name} in source"

    # Walk every Call node inside the method.
    httpx_post_calls = []
    for sub in ast.walk(target):
        if not isinstance(sub, ast.Call):
            continue
        f = sub.func
        # match `<name>_httpx_client.post(...)` calls
        if isinstance(f, ast.Attribute) and f.attr == "post":
            value = f.value
            if isinstance(value, ast.Name) and value.id.endswith("_httpx_client"):
                httpx_post_calls.append(sub)
    assert httpx_post_calls, (
        f"{method_name}: no `_httpx_client.post(...)` call sites found — "
        "test scaffolding may be stale"
    )
    for call in httpx_post_calls:
        kw_names = [kw.arg for kw in call.keywords]
        assert "logging_obj" in kw_names, (
            f"{method_name}: a `*_httpx_client.post(...)` site at line "
            f"{call.lineno} does not forward `logging_obj=` "
            f"(kwargs present: {kw_names}). This re-introduces LIT-1756."
        )


# ---- Behavioural regression via direct handler call ------------------------

@pytest.mark.asyncio
async def test_async_response_api_handler_records_llm_api_duration_ms() -> None:
    """End-to-end on ``async_response_api_handler``: stub the upstream httpx
    send and assert the decorator has set ``llm_api_duration_ms`` on the
    logging object after the handler returns. Before the fix this stayed
    ``None``; after the fix it must be a positive number.
    """
    from litellm.litellm_core_utils.litellm_logging import (
        Logging as LiteLLMLoggingObj,
    )
    from litellm.types.router import GenericLiteLLMParams
    from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse

    class StubResponsesAPIConfig:
        def validate_environment(self, headers, model, litellm_params):
            return headers or {}

        def get_complete_url(self, api_base, litellm_params):
            return f"{api_base}/responses"

        def transform_responses_api_request(
            self, model, input, response_api_optional_request_params,
            litellm_params, headers,
        ):
            return {"model": model, "input": input}

        def transform_response_api_response(self, model, raw_response, logging_obj):
            body = raw_response.json()
            return ResponsesAPIResponse(
                id=body["id"], object=body["object"],
                created_at=body["created_at"], model=body["model"],
                status=body["status"], output=body["output"],
                usage=ResponseAPIUsage(**body["usage"]),
            )

        def should_fake_stream(self, *a, **k):
            return False

    logging_obj = LiteLLMLoggingObj(
        model="gpt-4o", messages=[], stream=False, call_type="aresponses",
        start_time=None, litellm_call_id="test-call-id", function_id="fn",
    )
    logging_obj.model_call_details = {}

    handler = AsyncHTTPHandler()

    async def fake_send(self, request, **_kwargs):
        await asyncio.sleep(0.05)
        return httpx.Response(
            200, json=_fake_responses_payload(), request=request
        )

    base = BaseLLMHTTPHandler()
    with patch.object(httpx.AsyncClient, "send", fake_send):
        await base.async_response_api_handler(
            model="gpt-4o", input="hi",
            responses_api_provider_config=StubResponsesAPIConfig(),
            response_api_optional_request_params={},
            custom_llm_provider="openai",
            litellm_params=GenericLiteLLMParams(
                api_base="http://stub.invalid", api_key="sk-x"
            ),
            logging_obj=logging_obj,
            client=handler,
        )

    duration = logging_obj.model_call_details.get("llm_api_duration_ms")
    assert duration is not None, (
        "llm_api_duration_ms not set on logging_obj after handler call — "
        "track_llm_api_timing did not see logging_obj because it was not "
        "forwarded to post() (LIT-1756)"
    )
    assert isinstance(duration, (int, float)) and duration > 0, (
        f"llm_api_duration_ms must be a positive number, got {duration!r}"
    )
