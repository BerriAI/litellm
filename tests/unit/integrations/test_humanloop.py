import asyncio
import importlib
import os
from collections.abc import AsyncIterator, Mapping
from typing import Final

import httpx
import pytest
import pytest_asyncio

import litellm
from litellm.constants import LOGGING_WORKER_MAX_TIME_PER_COROUTINE
from litellm.integrations import humanloop
from litellm.integrations.humanloop import HumanLoopPromptManager
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome


def test_compile_prompt():
    prompt_manager = HumanLoopPromptManager()
    prompt_template = [
        {
            "content": "You are {{person}}. Answer questions as this person. Do not break character.",
            "name": None,
            "tool_call_id": None,
            "role": "system",
            "tool_calls": None,
        }
    ]
    prompt_variables = {"person": "John"}
    compiled_prompt = prompt_manager._compile_prompt_helper(prompt_template, prompt_variables)
    assert compiled_prompt[0]["content"] == "You are John. Answer questions as this person. Do not break character."


def _serve_humanloop_prompt(monkeypatch: pytest.MonkeyPatch, prompt_body: Mapping[str, object]) -> None:
    def humanloop_api(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v5/prompts/pr_1" and request.headers.get("X-Api-Key") == "hl-key":
            return httpx.Response(200, json=prompt_body)
        return httpx.Response(404, json={"detail": "not found"})

    monkeypatch.setattr(
        humanloop,
        "get_httpx_client",
        lambda: HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(humanloop_api))),
    )


@pytest.mark.parametrize(
    ("template", "expected_template"),
    [
        ({"role": "system", "content": "You are {{person}}."}, [{"role": "system", "content": "You are {{person}}."}]),
        (
            [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "{{question}}"}],
            [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "{{question}}"}],
        ),
    ],
)
def test_humanloop_prompt_from_api_lists_its_template_and_keeps_only_openai_params(
    monkeypatch: pytest.MonkeyPatch, template: object, expected_template: list[dict[str, str]]
) -> None:
    _serve_humanloop_prompt(
        monkeypatch,
        {
            "id": "pr_1",
            "name": "support-bot",
            "template": template,
            "model": "gpt-4o",
            "temperature": 0.3,
            "max_tokens": 64,
        },
    )

    prompt: Final = HumanLoopPromptManager()._get_prompt_from_id_api("pr_1", "hl-key")

    assert prompt["prompt_template"] == expected_template
    assert prompt["model"] == "gpt-4o"
    assert prompt["optional_params"] == {"temperature": 0.3, "max_tokens": 64}


def test_humanloop_prompt_from_api_rejects_a_template_that_is_not_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve_humanloop_prompt(monkeypatch, {"id": "pr_1", "template": "plain text", "model": "gpt-4o"})

    with pytest.raises(ValueError, match="Invalid template message type: <class 'str'>"):
        HumanLoopPromptManager()._get_prompt_from_id_api("pr_1", "hl-key")


@pytest.fixture(autouse=True)
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)


@pytest_asyncio.fixture(loop_scope="function", autouse=True)
async def drain_logging_worker(isolate_litellm_state: None) -> AsyncIterator[None]:
    yield
    await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=LOGGING_WORKER_DRAIN_TIMEOUT_SECONDS)


LOGGING_WORKER_DRAIN_TIMEOUT_SECONDS: Final = LOGGING_WORKER_MAX_TIME_PER_COROUTINE + 5.0


@pytest.fixture(scope="function", autouse=True)
def isolate_litellm_state():
    """
    Per-function isolation fixture.

    Resets litellm state to the true defaults captured at conftest import time,
    then restores after the test. This prevents module-level mutations (e.g.
    `litellm.num_retries = 3` at the top of test_langfuse_e2e_test.py) from
    leaking across tests within the same xdist worker.
    """
    from litellm.litellm_core_utils import litellm_logging as ll_logging
    from litellm.proxy.management_helpers import audit_logs as ll_audit_logs

    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()
    ll_logging._in_memory_loggers.clear()
    ll_audit_logs._audit_log_callback_cache.clear()
    for attr in _LIST_ATTRS:
        if attr in _DEFAULTS:
            default = _DEFAULTS[attr]
            setattr(litellm, attr, default.copy() if isinstance(default, list) else default)
    for attr in _SCALAR_ATTRS:
        if attr in _DEFAULTS:
            setattr(litellm, attr, _DEFAULTS[attr])
    yield
    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()
    ll_logging._in_memory_loggers.clear()
    ll_audit_logs._audit_log_callback_cache.clear()
    for attr in _LIST_ATTRS:
        if attr in _DEFAULTS:
            default = _DEFAULTS[attr]
            setattr(litellm, attr, default.copy() if isinstance(default, list) else default)
    for attr in _SCALAR_ATTRS:
        if attr in _DEFAULTS:
            setattr(litellm, attr, _DEFAULTS[attr])


_LIST_ATTRS = (
    "callbacks",
    "success_callback",
    "failure_callback",
    "_async_success_callback",
    "_async_failure_callback",
    "service_callback",
    "pre_call_rules",
    "post_call_rules",
)

_SCALAR_ATTRS = (
    "set_verbose",
    "cache",
    "num_retries",
    "num_retries_per_request",
    "turn_off_message_logging",
    "redact_messages_in_exceptions",
    "redact_user_api_key_info",
    "s3_callback_params",
    "s3_audit_callback_params",
    "datadog_params",
    "vector_store_registry",
)

_DEFAULTS: dict = {}


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    """
    Module-scoped setup. Reloads litellm only in single-process mode
    (skipped under xdist to avoid cross-worker interference).
    """
    import litellm

    worker_id = os.environ.get("PYTEST_XDIST_WORKER", None)
    if worker_id is None:
        importlib.reload(litellm)
        try:
            if hasattr(litellm, "proxy") and hasattr(litellm.proxy, "proxy_server"):
                import litellm.proxy.proxy_server

                importlib.reload(litellm.proxy.proxy_server)
        except Exception:
            pass
        if hasattr(litellm, "in_memory_llm_clients_cache"):
            litellm.in_memory_llm_clients_cache.flush_cache()
    yield
