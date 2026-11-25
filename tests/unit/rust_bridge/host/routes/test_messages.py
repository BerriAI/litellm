import contextlib
import datetime
import errno
import functools
import json
import logging
import os
import queue
import socket
import ssl
import struct
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import MappingProxyType
from typing import Final

import httpx
import pytest
from google.auth import exceptions as google_auth
from pydantic import TypeAdapter, ValidationError

import litellm
from litellm import anthropic_beta_headers_manager
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.constants import (
    ANTHROPIC_MESSAGES_DETACHED_STREAM_DRAIN_TIMEOUT_SECONDS,
    ANTHROPIC_MESSAGES_MAX_DETACHED_STREAM_DRAINS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
)
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.model_latest_aliases import get_model_latest_aliases
from litellm.llms.anthropic.messages.native import interceptors, mcp_handler
from litellm.llms.anthropic.messages.native.interceptors.base import MessagesInterceptor
from litellm.llms.anthropic.messages.native.utils import anthropic_messages_optional_param_keys
from litellm.llms.azure.common_utils import AzureOpenAIError, EntraRejectionError
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    exclude_rsa_key_exchange_and_chacha20,
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # its params dict is untyped
)
from litellm.rust_bridge import _native
from litellm.rust_bridge.host import callbacks_legacy_python
from litellm.rust_bridge.host.routes import messages as messages_host
from litellm.rust_bridge.host.routes.messages import response


def test_response_is_a_detached_public_messages_dict() -> None:
    native: Final = MappingProxyType(
        {
            "id": "msg_native",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [{"type": "text", "text": "native"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 2, "output_tokens": 3},
        }
    )

    built: Final = response(native)

    assert built == dict(native)
    assert isinstance(built, dict)
    built["_hidden_params"] = {"annotated": True}
    assert "_hidden_params" not in native


@pytest.mark.parametrize(
    ("disabled", "joined"),
    [pytest.param("true", False, id="httpx_transport"), pytest.param("false", True, id="aiohttp_transport")],
)
def test_settings_join_buffered_stream_reads_only_on_the_aiohttp_transport(
    monkeypatch: pytest.MonkeyPatch, disabled: str, joined: bool
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", False)
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", disabled)

    assert messages_host.settings({})["joined_reads"] is joined


@pytest.mark.parametrize(("asynchronous", "drains"), [(True, True), (False, False)])
def test_settings_drain_a_detached_stream_only_on_an_async_call(asynchronous: bool, drains: bool) -> None:
    projected: Final = messages_host.settings({}, asynchronous=asynchronous)

    assert (projected["detached_drain_timeout"], projected["detached_drain_limit"]) == (
        (ANTHROPIC_MESSAGES_DETACHED_STREAM_DRAIN_TIMEOUT_SECONDS, ANTHROPIC_MESSAGES_MAX_DETACHED_STREAM_DRAINS)
        if drains
        else (None, None)
    )


@pytest.mark.parametrize("asynchronous", [True, False])
def test_settings_retry_a_failed_connect_only_on_an_async_call(asynchronous: bool) -> None:
    assert messages_host.settings({}, asynchronous=asynchronous)["connect_retry"] is asynchronous


def test_settings_project_caller_configuration_without_resolving_a_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    monkeypatch.setattr(litellm, "reasoning_auto_summary", True)
    monkeypatch.setattr(litellm, "request_timeout", 1800)
    monkeypatch.setattr(litellm, "request_timeout_explicitly_set", True)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", False)
    monkeypatch.delenv("DISABLE_AIOHTTP_TRANSPORT", raising=False)

    projected: Final = messages_host.settings({"drop_params": "true", "additional_drop_params": ["metadata.user_id"]})

    assert projected == {
        "drop_params": True,
        "reasoning_auto_summary": True,
        "additional_drop_params": ("metadata.user_id",),
        "request_timeout": 1800.0,
        "drops_unsupported_params": True,
        "drops_rejected_safeguards": True,
        "model": None,
        "mock_response": None,
        "stream_cost_model": None,
        "extra_headers": None,
        "mcp_gateway": False,
        "ascii_response_headers": True,
        "joined_reads": True,
        "ascii_request_headers": True,
        "detached_drain_timeout": None,
        "detached_drain_limit": None,
        "connect_retry": False,
        "pricing": None,
    }


@pytest.mark.parametrize(
    ("global_drop_params", "kwargs", "drop_params", "drops_unsupported_params"),
    [
        pytest.param(True, {}, True, True, id="global-true"),
        pytest.param(1, {}, True, False, id="global-truthy-non-bool"),
        pytest.param(False, {"drop_params": "True"}, True, True, id="kwarg-flag-string"),
        pytest.param(False, {"drop_params": 1}, True, True, id="kwarg-one"),
        pytest.param(False, {"drop_params": "maybe"}, False, False, id="kwarg-not-a-flag"),
        pytest.param(False, {}, False, False, id="unset"),
    ],
)
def test_strict_tools_retry_reads_drop_params_as_the_python_retry_check_does(
    monkeypatch: pytest.MonkeyPatch,
    global_drop_params: object,
    kwargs: Mapping[str, object],
    drop_params: bool,
    drops_unsupported_params: bool,
) -> None:
    monkeypatch.setattr(litellm, "drop_params", global_drop_params)

    projected: Final = messages_host.settings(kwargs)

    assert projected["drop_params"] is drop_params
    assert projected["drops_unsupported_params"] is drops_unsupported_params


@pytest.mark.parametrize(
    ("timeout", "explicitly_set", "expected"),
    [
        (DEFAULT_REQUEST_TIMEOUT_SECONDS, False, None),
        (DEFAULT_REQUEST_TIMEOUT_SECONDS, True, DEFAULT_REQUEST_TIMEOUT_SECONDS),
        (300, False, 300.0),
    ],
)
def test_request_timeout_carries_only_a_configured_global_timeout(
    monkeypatch: pytest.MonkeyPatch, timeout: float, explicitly_set: bool, expected: float | None
) -> None:
    monkeypatch.setattr(litellm, "request_timeout", timeout)
    monkeypatch.setattr(litellm, "request_timeout_explicitly_set", explicitly_set)

    assert messages_host.settings({})["request_timeout"] == expected


@pytest.mark.parametrize(
    ("global_flag", "kwargs", "expected"),
    [
        (False, {}, False),
        (True, {}, True),
        (False, {"drop_params": "true"}, True),
        (False, {"drop_params": "nonsense"}, False),
        (False, {"drop_params": False}, False),
    ],
)
def test_drop_params_merges_the_global_flag_with_the_request(
    monkeypatch: pytest.MonkeyPatch, global_flag: bool, kwargs: dict[str, object], expected: bool
) -> None:
    monkeypatch.setattr(litellm, "drop_params", global_flag)

    assert messages_host.settings(kwargs)["drop_params"] is expected


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (["tools[*].input_examples", 3, "metadata.user_id"], ("tools[*].input_examples", "metadata.user_id")),
        ("tools", ()),
        (None, ()),
    ],
)
def test_additional_drop_params_keep_only_string_paths(configured: object, expected: tuple[str, ...]) -> None:
    settings: Final = messages_host.settings({"additional_drop_params": configured})

    assert settings["additional_drop_params"] == expected


def test_native_request_rejections_map_to_the_public_400() -> None:
    from types import MappingProxyType

    from litellm.rust_bridge.public_call import NativeCall

    request: Final = NativeCall(
        args=(),
        kwargs=MappingProxyType({}),
        base={
            "model": "anthropic/claude-sonnet-5",
            "messages": (),
            "max_tokens": 8,
            "stream": None,
            "api_key": None,
            "api_base": None,
            "custom_llm_provider": None,
            **MappingProxyType({}),
        },
    )
    rejected: Final = ValueError("claude-sonnet-5 does not support top_k=5")
    rejected.messages_request_error = True  # pyright: ignore[reportAttributeAccessIssue]  # marker the native host sets

    mapped: Final = messages_host.map_failure(rejected, request.resolved, "anthropic")

    assert isinstance(mapped, litellm.BadRequestError)
    assert mapped.status_code == 400
    assert "does not support top_k=5" in mapped.message
    assert mapped.model == "claude-sonnet-5"
    assert not isinstance(
        messages_host.map_failure(ValueError("plain"), request.resolved, "anthropic"), litellm.BadRequestError
    )


@pytest.mark.parametrize("custom_llm_provider", (None, "anthropic"), ids=("provider-in-the-model", "provider-passed"))
def test_a_rejection_before_the_send_names_the_resolved_provider_and_is_raised_as_python_raises_it(
    custom_llm_provider: str | None,
) -> None:
    rejected: Final = ValueError("messages: no message has content left after dropping empty or non-string text blocks")
    rejected.messages_request_error = True  # pyright: ignore[reportAttributeAccessIssue]  # marker the native host sets
    rejected.messages_unresolved_rejection = True  # pyright: ignore[reportAttributeAccessIssue]  # marker the native host sets

    mapped: Final = messages_host.map_failure(
        rejected,
        {"model": "anthropic/claude-sonnet-5", "custom_llm_provider": custom_llm_provider, "messages": []},
        "anthropic",
    )

    assert isinstance(mapped, litellm.BadRequestError)
    assert mapped.status_code == 400
    assert (mapped.model, mapped.llm_provider) == ("anthropic/claude-sonnet-5", "anthropic")
    assert mapped.__traceback__ is not None


def test_stream_hidden_params_projects_upstream_headers_the_way_the_python_handler_does() -> None:
    hidden: Final = messages_host.stream_hidden_params(
        (("request-id", "req_upstream_123"), ("x-ratelimit-remaining-requests", "41"))
    )

    additional: Final = hidden["additional_headers"]
    assert isinstance(additional, dict)
    assert additional["llm_provider-request-id"] == "req_upstream_123"
    assert additional["x-ratelimit-remaining-requests"] == "41"
    assert additional["request-id"] == additional["llm_provider-request-id"]


@pytest.mark.parametrize("relayed", [False, True])
def test_stream_hidden_params_carry_whether_the_native_relay_serves_the_stream(relayed: bool) -> None:
    from litellm.rust_bridge.messages.entrypoints import MessagesStreamHead

    head: Final = messages_host.stream_hidden_params((), relayed)

    assert isinstance(head, MessagesStreamHead)
    assert head.relayed is relayed
    assert "relayed" not in head


_ADAPTIVE_OPUS: Final = "anthropic/claude-opus-4-7"
_BEDROCK_INVOKE: Final = "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0"
_AWS: Final = MappingProxyType(
    {"aws_access_key_id": "test-access-key", "aws_secret_access_key": "test-secret-key", "aws_region_name": "us-east-1"}
)


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        pytest.param({"model": "azure_ai/claude-sonnet-4-5"}, "", id="no_token_is_empty"),
        pytest.param(
            {"model": "azure_ai/claude-sonnet-4-5", "azure_ad_token": "entra-token"}, "entra-token", id="static_token"
        ),
        pytest.param(
            {
                "model": "azure_ai/claude-sonnet-4-5",
                "azure_ad_token": "entra-token",
                "azure_ad_token_provider": lambda: "minted",
            },
            "minted",
            id="provider_wins",
        ),
    ],
)
def test_azure_ad_token_is_the_one_python_mints(
    arguments: dict[str, object], expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AZURE_AD_TOKEN", raising=False)
    monkeypatch.setattr(litellm, "enable_azure_ad_token_refresh", False)
    assert messages_host.azure_ad_token(arguments) == expected


@pytest.mark.parametrize(
    ("model", "provider", "expected"),
    [
        pytest.param("anthropic/claude-sonnet-4-5", "anthropic", False, id="anthropic"),
        pytest.param("azure_ai/claude-sonnet-4-5", "azure_ai", True, id="azure_ai"),
        pytest.param(_BEDROCK_INVOKE, "bedrock", True, id="bedrock_invoke"),
    ],
)
def test_filters_beta_headers_is_what_the_provider_config_says(model: str, provider: str, expected: bool) -> None:
    assert messages_host.filters_beta_headers(model, provider, None) is expected


def test_the_beta_map_stays_unloaded_until_a_filter_asks_and_then_loads_off_the_calling_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote: Final[dict[str, object]] = {"provider_aliases": {}, "azure_ai": {"advisor-tool-2026-03-01": None}}
    fetched_on: Final = queue.SimpleQueue[str]()
    release: Final = threading.Event()

    def fetch(url: str) -> dict[str, object]:
        fetched_on.put(threading.current_thread().name)
        assert release.wait(timeout=10)
        return remote

    monkeypatch.setattr(anthropic_beta_headers_manager, "_BETA_HEADERS_CONFIG", None)
    monkeypatch.setattr(anthropic_beta_headers_manager, "get_beta_headers_config", fetch)

    assert messages_host.loaded_beta_headers_config() is None
    assert fetched_on.empty()

    messages_host.load_beta_headers_config_in_background()

    assert fetched_on.get(timeout=10) != threading.current_thread().name
    assert messages_host.loaded_beta_headers_config() is None
    release.set()
    assert anthropic_beta_headers_manager.get_provider_name("azure_ai") == "azure_ai"
    assert messages_host.loaded_beta_headers_config() is remote
    assert fetched_on.empty()


def test_repeated_load_requests_during_a_slow_download_queue_a_single_load(monkeypatch: pytest.MonkeyPatch) -> None:
    release: Final = threading.Event()
    submitted: Final = queue.SimpleQueue[str]()
    loader: Final = ThreadPoolExecutor(max_workers=1)

    class RecordingLoader:
        def submit(self, load: Callable[[], None]) -> Future[None]:
            submitted.put(load.__name__)
            return loader.submit(load)

    def fetch(url: str) -> dict[str, object]:
        assert release.wait(timeout=10)
        return {"provider_aliases": {}}

    monkeypatch.setattr(anthropic_beta_headers_manager, "_BETA_HEADERS_CONFIG", None)
    monkeypatch.setattr(anthropic_beta_headers_manager, "get_beta_headers_config", fetch)
    monkeypatch.setattr(anthropic_beta_headers_manager, "_BETA_HEADERS_CONFIG_LOADER", RecordingLoader())

    for _ in range(20):
        messages_host.load_beta_headers_config_in_background()
    release.set()
    loader.shutdown(wait=True)

    assert submitted.qsize() == 1
    assert messages_host.loaded_beta_headers_config() is not None


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        pytest.param({"model": "azure_ai/claude-sonnet-4-5"}, "azure_ai", id="model_prefix"),
        pytest.param(
            {"model": "claude-sonnet-4-5", "custom_llm_provider": "azure_ai"}, "azure_ai", id="explicit_provider"
        ),
        pytest.param({"model": _BEDROCK_INVOKE}, "bedrock", id="bedrock_prefix"),
        pytest.param(
            {"model": "us.anthropic.claude-sonnet-4-5-20250929-v1:0", "custom_llm_provider": "bedrock"},
            "bedrock",
            id="bedrock_explicit_provider",
        ),
        pytest.param({"model": "claude-sonnet-4-5"}, "anthropic", id="anthropic_model"),
        pytest.param({"model": "not-a-known-model"}, "anthropic", id="unresolved"),
    ],
)
def test_failure_provider_is_the_provider_python_maps_errors_for(call: dict[str, object], expected: str) -> None:
    assert messages_host.failure_provider(call) == expected


@pytest.mark.parametrize(
    ("model", "expected"),
    [("claude-sonnet-4-5", "anthropic"), ("gpt-4o", "openai"), ("not-a-known-model", None)],
)
def test_inferred_provider_is_the_one_the_python_handler_resolves(model: str, expected: str | None) -> None:
    assert messages_host.inferred_provider(model, None) == expected


@pytest.mark.parametrize(
    ("request_effort", "levels", "drop_params", "expected"),
    [
        ("max", ["low", "high"], False, "high"),
        ("high", ["low"], True, "low"),
    ],
)
def test_body_params_round_effort_to_the_levels_the_deployment_declares(
    request_effort: str, levels: list[str], drop_params: bool, expected: str
) -> None:
    request: Final[dict[str, object]] = {
        "model": _ADAPTIVE_OPUS,
        "messages": [],
        "max_tokens": 8,
        "output_config": {"effort": request_effort},
        "drop_params": drop_params,
        "model_info": {"id": "deployment-1", "reasoning_effort_levels": levels},
    }

    assert messages_host.body_params(request, asynchronous=True)["output_config"] == {"effort": expected}


def test_body_params_keep_the_requested_effort_without_deployment_levels() -> None:
    request: Final[dict[str, object]] = {
        "model": _ADAPTIVE_OPUS,
        "messages": [],
        "max_tokens": 8,
        "output_config": {"effort": "max"},
    }

    assert messages_host.body_params(request, asynchronous=True)["output_config"] == {"effort": "max"}


@pytest.mark.parametrize(
    ("additional_drop_params", "expected_keys"),
    [
        (["thinking"], {"max_tokens"}),
        (["top_k"], {"max_tokens", "thinking", "output_config"}),
    ],
)
def test_body_params_derive_no_effort_from_a_param_the_deployment_lists(
    additional_drop_params: list[str], expected_keys: set[str]
) -> None:
    request: Final[dict[str, object]] = {
        "model": _ADAPTIVE_OPUS,
        "messages": [],
        "max_tokens": 8192,
        "thinking": {"type": "enabled", "budget_tokens": 4096},
        "additional_drop_params": additional_drop_params,
        "model_info": {"id": "deployment-1", "reasoning_effort_levels": ["low", "medium", "high"]},
    }

    assert set(messages_host.body_params(request, asynchronous=True)).difference({"messages", "model"}) == expected_keys


@pytest.mark.parametrize(
    ("thread", "expected"),
    [
        ({"type": "create"}, {"type": "create"}),
        ({"type": "continue", "previous_message_id": "msg_1"}, {"type": "continue", "previous_message_id": "msg_1"}),
        (None, None),
    ],
)
def test_body_params_forward_the_thread_like_the_python_handler(
    thread: dict[str, str] | None, expected: dict[str, str] | None
) -> None:
    request: Final[dict[str, object]] = {
        "model": "claude-sonnet-4-5",
        "messages": [],
        "max_tokens": 8,
        "thread": thread,
    }

    assert messages_host.body_params(request, asynchronous=True).get("thread") == expected


@pytest.mark.parametrize(
    ("metadata", "expected"),
    (
        ({"user_id": "u-1", "user_api_key_auth": object(), "pillar_flagged": False}, {"user_id": "u-1"}),
        ({"user_api_key_auth": object()}, {}),
        (["u-1"], ["u-1"]),
    ),
    ids=("litellm-keys-dropped", "only-litellm-keys", "not-an-object-left-for-rust-to-reject"),
)
def test_body_params_forward_only_the_metadata_anthropic_accepts(metadata: object, expected: object) -> None:
    request: Final[dict[str, object]] = {
        "model": "claude-sonnet-4-5",
        "messages": [],
        "max_tokens": 8,
        "metadata": metadata,
    }

    assert messages_host.body_params(request, asynchronous=True)["metadata"] == expected


def test_body_params_keep_a_call_param_and_an_unknown_key_out_of_the_body() -> None:
    request: Final[dict[str, object]] = {
        "model": "claude-sonnet-4-5",
        "messages": [],
        "max_tokens": 8,
        "cache_control": {"type": "ephemeral"},
        "user": "end-user-1",
        "tags": ["a"],
        "num_retries": 2,
        "not_a_param": 1,
    }

    assert messages_host.body_params(request, asynchronous=True) == {
        "max_tokens": 8,
        "cache_control": {"type": "ephemeral"},
    }


def test_the_rust_route_reads_the_params_the_python_handler_keeps() -> None:
    assert frozenset(_native.messages_optional_params()) == anthropic_messages_optional_param_keys()


def _marked(phase: str | None = None, *args: object) -> ValueError:
    error: Final = ValueError(*args)
    if phase is not None:
        error.messages_timeout = phase  # pyright: ignore[reportAttributeAccessIssue]  # marker the native host sets
    return error


@pytest.mark.parametrize(
    ("disabled", "text"), ((None, "Timeout on reading data from socket"), ("true", "")), ids=("aiohttp", "httpx")
)
def test_a_stalled_stream_body_reaches_the_caller_as_the_transport_timeout(
    monkeypatch: pytest.MonkeyPatch, disabled: str | None, text: str
) -> None:
    import httpx

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", False)
    if disabled is None:
        monkeypatch.delenv("DISABLE_AIOHTTP_TRANSPORT", raising=False)
    else:
        monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", disabled)

    mapped: Final = messages_host.map_failure(_marked("Body"), {"model": _ADAPTIVE_OPUS}, "anthropic")

    assert isinstance(mapped, httpx.ReadTimeout)
    assert str(mapped) == text


@pytest.mark.parametrize(
    ("value", "received", "text"),
    (
        ("h\ud800".encode("utf-8", "surrogatepass"), False, "can't encode character '\\ud800' in position 1"),
        ("caf\u00e9".encode(), True, "can't encode character '\\xe9' in position 3"),
        (b"caf\xe9", True, "can't encode character '\\udce9' in position 3"),
    ),
    ids=("sent-surrogate", "received-utf-8", "received-latin-1"),
)
def test_an_unencodable_header_is_the_error_pythons_ascii_codec_raises(value: bytes, received: bool, text: str) -> None:
    error: Final = messages_host.unencodable_header(value, received)

    assert isinstance(error, UnicodeEncodeError)
    assert text in str(error)


def _flagged(marker: str) -> ValueError:
    error: Final = ValueError("Missing Azure API Base")
    setattr(error, marker, True)
    return error


@pytest.mark.parametrize(
    "mapped",
    (
        pytest.param(
            lambda: messages_host.map_failure(
                _flagged("messages_python_raised"), {"model": _ADAPTIVE_OPUS}, "azure_ai"
            ),
            id="python-raised",
        ),
        pytest.param(
            lambda: messages_host.map_failure(
                _flagged("messages_missing_credential"), {"model": _ADAPTIVE_OPUS}, "azure_ai"
            ),
            id="missing-credential",
        ),
        pytest.param(lambda: messages_host.azure_ad_token_failure(RuntimeError("provider")), id="token-provider"),
        pytest.param(lambda: messages_host.non_string_azure_ad_token(7), id="non-string-token"),
    ),
)
def test_a_failure_python_raises_before_the_send_is_logged_with_its_traceback(mapped: Callable[[], Exception]) -> None:
    from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup

    assert StandardLoggingPayloadSetup.get_error_information(mapped()).get("traceback")


@pytest.mark.parametrize("marker", ("messages_python_raised", "messages_missing_credential"))
def test_an_anthropic_failure_before_the_send_stays_unraised_as_on_main(marker: str) -> None:
    from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup

    mapped: Final = messages_host.map_failure(_flagged(marker), {"model": _ADAPTIVE_OPUS}, "anthropic")

    assert mapped.__traceback__ is None
    assert not StandardLoggingPayloadSetup.get_error_information(mapped).get("traceback")


def test_a_native_read_timeout_maps_to_the_public_timeout_with_python_text() -> None:
    mapped: Final = messages_host.map_failure(_marked("Read"), {"model": _ADAPTIVE_OPUS, "timeout": 1.0}, "anthropic")

    assert isinstance(mapped, litellm.Timeout)
    assert "Read timed out. Timeout passed=1.0" in str(mapped)


@pytest.mark.parametrize(("phase", "billed"), (("Read", True), ("Connection", False)))
def test_a_native_timeout_is_billed_on_the_failure_row_as_python_bills_it(phase: str, billed: bool) -> None:
    from litellm.proxy.utils import _provider_billed_the_failure

    mapped: Final = messages_host.map_failure(_marked(phase), {"model": _ADAPTIVE_OPUS, "timeout": 1.0}, "anthropic")

    assert isinstance(mapped, litellm.Timeout)
    assert _provider_billed_the_failure(mapped) is billed


def test_an_upstream_status_maps_through_the_provider_error_class() -> None:
    upstream: Final = ValueError(429, "slow down")
    upstream.headers = [("retry-after", "3")]  # pyright: ignore[reportAttributeAccessIssue]  # set by the native host

    mapped: Final = messages_host.map_failure(upstream, {"model": _ADAPTIVE_OPUS}, "anthropic")

    assert isinstance(mapped, litellm.RateLimitError)
    assert "slow down" in str(mapped)
    assert mapped.__context__ is upstream


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("", {"content-type": "text/plain"}),
        ("gateway", {"content-type": "text/plain", "content-length": "7"}),
    ],
)
def test_upstream_failure_headers_match_the_python_clients_rebuilt_response(
    body: str, expected: dict[str, str]
) -> None:
    from litellm.litellm_core_utils.exception_mapping_utils import _get_response_headers

    upstream: Final = ValueError(504, body)
    upstream.headers = [  # pyright: ignore[reportAttributeAccessIssue]  # set by the native host
        ("content-type", "text/plain"),
        ("content-encoding", "gzip"),
        ("content-length", "42"),
    ]

    mapped: Final = messages_host.map_failure(upstream, {"model": _ADAPTIVE_OPUS}, "anthropic")

    assert isinstance(mapped, litellm.Timeout)
    assert dict(_get_response_headers(mapped) or {}) == expected


@pytest.mark.parametrize("status", [200, 500])
def test_upstream_failure_without_wire_headers_states_no_headers(status: int) -> None:
    from litellm.litellm_core_utils.exception_mapping_utils import _get_response_headers

    upstream: Final = ValueError(status, "not a messages response")
    upstream.headers = []  # pyright: ignore[reportAttributeAccessIssue]  # set by the native host

    mapped: Final = messages_host.map_failure(upstream, {"model": _ADAPTIVE_OPUS}, "anthropic")

    assert isinstance(mapped, Exception)
    assert not dict(_get_response_headers(mapped) or {})


def test_a_late_rejection_tells_the_logger_the_resolved_call_first() -> None:
    import datetime

    from litellm.litellm_core_utils.litellm_logging import Logging

    logger: Final = Logging(
        model=_ADAPTIVE_OPUS,
        messages=[],
        stream=False,
        call_type="anthropic_messages",
        start_time=datetime.datetime.now(),
        litellm_call_id="late",
        function_id="late",
    )
    rejected: Final = ValueError("budget_tokens must be a number")
    rejected.messages_late_rejection = True  # pyright: ignore[reportAttributeAccessIssue]  # marker the native host sets
    rejected.messages_request_error = True  # pyright: ignore[reportAttributeAccessIssue]  # marker the native host sets

    mapped: Final = messages_host.map_failure(
        rejected,
        {"model": _ADAPTIVE_OPUS, "messages": [], "max_tokens": 8, "litellm_logging_obj": logger},
        "anthropic",
    )

    assert isinstance(mapped, litellm.BadRequestError)
    assert logger.model_call_details["custom_llm_provider"] == "anthropic"
    assert logger.model_call_details["optional_params"] == {"max_tokens": 8}


@pytest.mark.parametrize(
    ("aiohttp", "cut", "error_class", "text"),
    (
        (
            True,
            (60, None, False),
            "ReadError",
            "Response payload is not completed: <TransferEncodingError: 400, "
            "message='Not enough data to satisfy transfer length header.'>",
        ),
        (
            True,
            (19, 100, True),
            "ReadError",
            "Response payload is not completed: <ContentLengthError: 400, "
            "message='Not enough data to satisfy content length header (received 19 of 100 bytes).'>. "
            "ConnectionResetError(104, 'Connection reset by peer')",
        ),
        (
            False,
            (60, None, False),
            "RemoteProtocolError",
            "peer closed connection without sending complete message body (incomplete chunked read)",
        ),
        (
            False,
            (19, 100, False),
            "RemoteProtocolError",
            "peer closed connection without sending complete message body (received 19 bytes, expected 100)",
        ),
        (False, (19, 100, True), "ReadError", ""),
    ),
)
def test_a_stream_body_cut_reaches_the_caller_as_the_transport_raised_it(
    monkeypatch: pytest.MonkeyPatch,
    aiohttp: bool,
    cut: tuple[int, int | None, bool],
    error_class: str,
    text: str,
) -> None:
    import httpx

    from litellm.rust_bridge.host import http_transport

    monkeypatch.setattr(http_transport, "uses_aiohttp_transport", lambda: aiohttp)
    native: Final = RuntimeError("upstream network error")
    native.messages_body_cut = cut  # pyright: ignore[reportAttributeAccessIssue]  # marker the native host sets

    mapped: Final = messages_host.map_failure(native, {"model": _ADAPTIVE_OPUS, "stream": True}, "anthropic")

    assert type(mapped) is getattr(httpx, error_class)
    assert str(mapped) == text


@pytest.mark.parametrize(
    ("aiohttp", "cut", "text"),
    (
        (
            True,
            (19, 100, False),
            "Response payload is not completed: <ContentLengthError: 400, "
            "message='Not enough data to satisfy content length header (received 19 of 100 bytes).'>",
        ),
        (
            False,
            (19, 100, False),
            "peer closed connection without sending complete message body (received 19 bytes, expected 100)",
        ),
    ),
)
def test_a_complete_body_cut_is_a_synthesized_server_error(
    monkeypatch: pytest.MonkeyPatch,
    aiohttp: bool,
    cut: tuple[int, int | None, bool],
    text: str,
) -> None:
    from litellm.rust_bridge.host import http_transport

    monkeypatch.setattr(http_transport, "uses_aiohttp_transport", lambda: aiohttp)
    native: Final = RuntimeError("upstream network error")
    native.messages_body_cut = cut  # pyright: ignore[reportAttributeAccessIssue]  # marker the native host sets

    mapped: Final = messages_host.map_failure(native, {"model": _ADAPTIVE_OPUS}, "anthropic")

    assert type(mapped) is litellm.InternalServerError
    assert mapped.status_code == 500
    assert text in str(mapped)


def _deployment_logger() -> Logging:
    logger: Final = Logging(
        model=_ADAPTIVE_OPUS,
        messages=[],
        stream=True,
        call_type="anthropic_messages",
        start_time=datetime.datetime.now(),
        litellm_call_id="cut",
        function_id="cut",
    )
    logger.model_call_details["custom_llm_provider"] = "anthropic"
    logger.model_call_details["litellm_params"] = {
        "api_base": "https://upstream.example/v1",
        "metadata": {"model_group": "claude-group", "deployment": "anthropic/claude-opus-4-6"},
    }
    return logger


@pytest.mark.parametrize("marker", ("body_cut", "body_timeout"))
def test_a_stream_transport_error_names_the_deployment_like_the_python_stream_iterator(marker: str) -> None:
    native: Final = RuntimeError("upstream network error")
    if marker == "body_cut":
        native.messages_body_cut = (19, None, False)  # pyright: ignore[reportAttributeAccessIssue]  # marker the native host sets
    else:
        native.messages_timeout = "Body"  # pyright: ignore[reportAttributeAccessIssue]  # marker the native host sets
    request: Final = {"model": _ADAPTIVE_OPUS, "stream": True, "litellm_logging_obj": _deployment_logger()}

    debug_info: Final = getattr(messages_host.map_failure(native, request, "anthropic"), "litellm_debug_info", None)

    assert isinstance(debug_info, str)
    assert "API Base: `https://upstream.example/v1`" in debug_info
    assert "model_group: `claude-group`" in debug_info
    assert "deployment: `anthropic/claude-opus-4-6`" in debug_info


@pytest.mark.parametrize(
    ("provider", "named"),
    (
        pytest.param("azure_ai", True, id="azure_ai-like-the-python-stream-iterator"),
        pytest.param("anthropic", False, id="anthropic-bare-as-on-main"),
    ),
)
def test_a_stream_decoding_failure_names_the_deployment_only_for_a_provider_with_a_python_path(
    provider: str, named: bool
) -> None:
    request: Final = {"model": _ADAPTIVE_OPUS, "stream": True, "litellm_logging_obj": _deployment_logger()}

    failure: Final = messages_host.decoding_failure("Error -3 while decompressing data", request, provider)

    assert type(failure) is httpx.DecodingError
    assert ("deployment: `anthropic/claude-opus-4-6`" in str(getattr(failure, "litellm_debug_info", ""))) is named


_TRANSPORT: Final = "error sending request: client error"


@pytest.mark.parametrize(
    ("aiohttp", "native", "text"),
    (
        (
            True,
            f"could not reach the provider: {_TRANSPORT} (Connect): tcp connect error: Connection refused (os error 111)",
            "Cannot connect to host 127.0.0.1:9 ssl:<ssl> [Connect call failed ('127.0.0.1', 9)]",
        ),
        (
            True,
            f"could not reach the provider: {_TRANSPORT} (Connect): dns error: error resolving DNS: "
            "failed to lookup address information: Name or service not known",
            "Cannot connect to host 127.0.0.1:9 ssl:<ssl> [Name or service not known]",
        ),
        (
            True,
            f"could not reach the provider: {_TRANSPORT} (Connect): tls handshake eof",
            "Cannot connect to host 127.0.0.1:9 ssl:<ssl> [None]",
        ),
        (
            True,
            f"upstream network error: {_TRANSPORT} (SendRequest): connection error: Connection reset by peer (os error 104)",
            "[Errno 104] Connection reset by peer",
        ),
        (
            True,
            f"upstream network error: {_TRANSPORT} (SendRequest): connection closed before message completed",
            "Server disconnected",
        ),
        (
            False,
            f"could not reach the provider: {_TRANSPORT} (Connect): tcp connect error: Connection refused (os error 111)",
            "All connection attempts failed",
        ),
        (
            False,
            f"could not reach the provider: {_TRANSPORT} (Connect): dns error: error resolving DNS: "
            "failed to lookup address information: Name or service not known",
            "[Errno -2] Name or service not known",
        ),
        (False, f"could not reach the provider: {_TRANSPORT} (Connect): tls handshake eof", ""),
        (
            False,
            f"upstream network error: {_TRANSPORT} (SendRequest): connection error: Connection reset by peer (os error 104)",
            "",
        ),
        (
            False,
            f"upstream network error: {_TRANSPORT} (SendRequest): connection closed before message completed",
            "Server disconnected without sending a response.",
        ),
    ),
)
def test_a_failure_before_any_response_maps_like_the_python_transport_error(
    monkeypatch: pytest.MonkeyPatch, aiohttp: bool, native: str, text: str
) -> None:
    from litellm.rust_bridge.host import http_transport

    monkeypatch.setattr(http_transport, "uses_aiohttp_transport", lambda: aiohttp)
    monkeypatch.setattr(messages_host, "get_ssl_configuration", lambda _ssl_verify: "<ssl>")

    mapped: Final = messages_host.map_failure(
        RuntimeError(native), {"model": _ADAPTIVE_OPUS, "api_base": "http://127.0.0.1:9"}, "anthropic"
    )

    assert type(mapped) is litellm.InternalServerError
    assert mapped.status_code == 500
    assert mapped.message == (
        f"litellm.InternalServerError: AnthropicException - {text}. Handle with `litellm.InternalServerError`."
    )
    assert "This looks like a bug in LiteLLM" not in str(mapped)


_AZURE_CLAUDE: Final = "azure_ai/claude-sonnet-4-5"


def _provider_and_model(error: Exception) -> tuple[object, object]:
    return getattr(error, "llm_provider", None), getattr(error, "model", None)


def _ssl_configuration(_ssl_verify: object) -> str:
    return "<ssl>"


_REFUSED: Final = (
    f"could not reach the provider: {_TRANSPORT} (Connect): tcp connect error: Connection refused (os error 111)"
)


@pytest.mark.parametrize(
    ("aiohttp", "text"),
    (
        pytest.param(
            True, "Cannot connect to host 127.0.0.1:47 ssl:<ssl> [Connect call failed ('127.0.0.1', 47)]", id="aiohttp"
        ),
        pytest.param(False, "All connection attempts failed", id="httpx"),
    ),
)
def test_an_azure_ai_connect_failure_names_the_environment_base_the_host_sent_to(
    monkeypatch: pytest.MonkeyPatch, aiohttp: bool, text: str
) -> None:
    from litellm.rust_bridge.host import http_transport

    monkeypatch.setattr(http_transport, "uses_aiohttp_transport", lambda: aiohttp)
    monkeypatch.setattr(messages_host, "get_ssl_configuration", _ssl_configuration)

    mapped: Final = messages_host.map_failure(
        RuntimeError(_REFUSED),
        {"model": _AZURE_CLAUDE},
        "azure_ai",
        secrets_read=[("AZURE_API_BASE", "http://127.0.0.1:47/anthropic")],
    )

    assert type(mapped) is litellm.InternalServerError
    assert (mapped.status_code, *_provider_and_model(mapped)) == (500, "azure_ai", _AZURE_CLAUDE)
    assert mapped.message.endswith(f"Azure_aiException - {text}")


_UNSENDABLE_HEADER: Final = "invalid request: failed to parse header value"


@pytest.mark.parametrize(
    ("aiohttp", "text"),
    (
        pytest.param(
            True, "Forbidden control character detected in headers. Potential header injection attack.", id="aiohttp"
        ),
        pytest.param(False, _UNSENDABLE_HEADER, id="httpx"),
    ),
)
def test_an_unsendable_azure_ai_header_value_fails_as_the_python_transport_refuses_it(
    monkeypatch: pytest.MonkeyPatch, aiohttp: bool, text: str
) -> None:
    from litellm.rust_bridge.host import http_transport

    monkeypatch.setattr(http_transport, "uses_aiohttp_transport", lambda: aiohttp)

    mapped: Final = messages_host.map_failure(
        ValueError(_UNSENDABLE_HEADER), {"model": _AZURE_CLAUDE, "api_base": "http://127.0.0.1:9/anthropic"}, "azure_ai"
    )

    assert type(mapped) is litellm.InternalServerError
    assert (mapped.status_code, *_provider_and_model(mapped)) == (500, "azure_ai", _AZURE_CLAUDE)
    assert mapped.message.endswith(f"Azure_aiException - {text}")


@pytest.mark.parametrize("aiohttp", (True, False), ids=("aiohttp", "httpx"))
def test_an_unsendable_anthropic_header_value_stays_main_s_connection_error(
    monkeypatch: pytest.MonkeyPatch, aiohttp: bool
) -> None:
    from litellm.litellm_core_utils.error_normalization import PROVIDER_CONNECTION_ERROR, normalize_error
    from litellm.rust_bridge.host import http_transport

    monkeypatch.setattr(http_transport, "uses_aiohttp_transport", lambda: aiohttp)

    mapped: Final = messages_host.map_failure(
        ValueError(_UNSENDABLE_HEADER),
        {"model": "claude-sonnet-4-5", "api_base": "http://127.0.0.1:9/anthropic"},
        "anthropic",
    )

    assert type(mapped) is litellm.APIConnectionError
    assert (mapped.status_code, *_provider_and_model(mapped)) == (500, "anthropic", "claude-sonnet-4-5")
    assert str(mapped).startswith(f"litellm.APIConnectionError: {_UNSENDABLE_HEADER}\n")
    assert "https://github.com/BerriAI/litellm/issues/new?template=bug_report.yml" in str(mapped)
    assert normalize_error(mapped, "500", str(mapped)) == PROVIDER_CONNECTION_ERROR


def _upstream_status(status: int, body: str) -> Exception:
    upstream: Final = ValueError(status, body)
    upstream.headers = []  # pyright: ignore[reportAttributeAccessIssue]  # set by the native host
    return upstream


@pytest.mark.parametrize(
    "native",
    (
        pytest.param(_upstream_status(404, "not found"), id="upstream-status"),
        pytest.param(RuntimeError(_REFUSED), id="transport"),
    ),
)
@pytest.mark.parametrize(
    ("provider", "debug_info"),
    (
        pytest.param("azure_ai", f"\nModel: {_AZURE_CLAUDE}", id="azure_ai-like-the-python-entry"),
        pytest.param(
            "anthropic",
            f"\nModel: {_AZURE_CLAUDE}\nAPI Base: `http://127.0.0.1:9/anthropic`",
            id="anthropic-without-a-python-path",
        ),
    ),
)
def test_a_mapped_failure_names_the_call_s_api_base_only_for_a_provider_without_a_python_path(
    native: Exception, provider: str, debug_info: str
) -> None:
    mapped: Final = messages_host.map_failure(
        native, {"model": _AZURE_CLAUDE, "api_base": "http://127.0.0.1:9/anthropic", "api_key": "k"}, provider
    )

    assert getattr(mapped, "litellm_debug_info", None) == debug_info


_CALL: Final = MappingProxyType({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8})
_OPENCODE: Final = "https://opencode.ai/inference/go/anthropic/v1"
_COST_HEADER: Final = MappingProxyType({"headers": {"x-litellm-include-cost-in-streaming-usage": "true"}})


@pytest.mark.parametrize(
    ("model", "custom_llm_provider", "expected"),
    [
        (
            "anthropic/claude-opus-latest",
            None,
            f"anthropic/{get_model_latest_aliases()['anthropic']['claude-opus-latest']}",
        ),
        ("claude-opus-latest", "anthropic", get_model_latest_aliases()["anthropic"]["claude-opus-latest"]),
        ("anthropic/claude-sonnet-4-5", None, "anthropic/claude-sonnet-4-5"),
    ],
)
def test_settings_resolve_a_latest_alias_keeping_the_provider_prefix(
    model: str, custom_llm_provider: str | None, expected: str
) -> None:
    settings: Final = messages_host.settings({}, {**_CALL, "model": model, "custom_llm_provider": custom_llm_provider})

    assert settings["model"] == expected


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        ({"api_base": _OPENCODE, "session_id": "s-1"}, {"x-opencode-session": "s-1"}),
        (
            {"api_base": _OPENCODE, "session_id": "s-1", "headers": {"x-trace": "t"}},
            {"x-trace": "t", "x-opencode-session": "s-1"},
        ),
        ({"api_base": _OPENCODE, "session_id": "s-1", "extra_headers": {"x-opencode-session": "mine"}}, None),
        ({"api_base": "https://other.example/v1", "session_id": "s-1"}, None),
        ({"api_base": _OPENCODE, "session_id": "s-1", "mock_response": "hi"}, None),
    ],
)
def test_settings_add_only_the_affinity_header_the_caller_did_not_send(
    extra: dict[str, object], expected: dict[str, str] | None
) -> None:
    settings: Final = messages_host.settings({}, {**_CALL, "model": "anthropic/claude-sonnet-4-5", **extra})

    assert settings["extra_headers"] == expected


@pytest.mark.parametrize("variable", ["ANTHROPIC_API_BASE", "ANTHROPIC_BASE_URL"])
def test_settings_add_the_affinity_header_for_an_anthropic_base_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, variable: str
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_BASE", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setenv(variable, _OPENCODE)

    settings: Final = messages_host.settings({}, {**_CALL, "model": "anthropic/claude-sonnet-4-5", "session_id": "s-1"})

    assert settings["extra_headers"] == {"x-opencode-session": "s-1"}


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        ({"stream": True, "proxy_server_request": dict(_COST_HEADER)}, "claude-sonnet-4-5"),
        ({"proxy_server_request": dict(_COST_HEADER)}, None),
        ({"stream": True}, None),
        ({"stream": True, "proxy_server_request": dict(_COST_HEADER), "mock_response": "hi"}, None),
    ],
)
def test_settings_name_the_cost_model_only_for_an_opted_in_stream(
    extra: dict[str, object], expected: str | None
) -> None:
    settings: Final = messages_host.settings({}, {**_CALL, "model": "anthropic/claude-sonnet-4-5", **extra})

    assert settings["stream_cost_model"] == expected


def test_settings_carry_the_mock_the_python_handler_returns() -> None:
    settings: Final = messages_host.settings(
        {}, {**_CALL, "model": "anthropic/claude-sonnet-4-5", "mock_response": "hi"}
    )

    assert isinstance(settings["mock_response"], dict)
    assert settings["mock_response"]["content"] == [{"type": "text", "text": "hi"}]


def test_settings_infer_no_provider_off_an_affinity_host(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected(model: str, api_base: str | None) -> str | None:
        pytest.fail("the provider is only inferred for an affinity api_base")

    monkeypatch.setattr(messages_host, "inferred_provider", unexpected)

    settings: Final = messages_host.settings({}, {**_CALL, "model": "claude-sonnet-4-5"})

    assert settings["model"] == "claude-sonnet-4-5"
    assert settings["extra_headers"] is None


def test_body_params_leave_the_messages_to_the_route_for_a_call_without_prompt_caching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", False)

    fields: Final = messages_host.body_params(
        {
            "model": "anthropic/claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "hi"}],
            "system": "Be terse.",
            "enable_prompt_caching": False,
            "cache_control_injection_points": [],
        },
        asynchronous=True,
    )

    assert fields == {"system": "Be terse."}


@pytest.mark.parametrize("asynchronous", [True, False])
def test_body_params_mark_the_turns_tools_and_deployment_the_handler_marks(
    monkeypatch: pytest.MonkeyPatch, asynchronous: bool
) -> None:
    monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", False)
    gateway: Final[dict[str, object]] = {}
    caller_metadata: Final = {"user_id": "u"}
    tool: Final = {"name": "get_weather", "input_schema": {"type": "object"}}

    fields: Final = messages_host.body_params(
        {
            "model": "anthropic/claude-sonnet-4-5",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [{"type": "text", "text": ""}]},
                {"role": "user", "content": "again"},
            ],
            "system": "Be terse.",
            "tools": [tool],
            "metadata": caller_metadata,
            "litellm_metadata": gateway,
            "model_info": {"id": "deployment-1"},
            "cache_control_injection_points": [
                {"location": "message", "role": "system"},
                {"location": "message", "index": -1, "control": {"type": "ephemeral", "ttl": "1h"}},
                {"location": "tool_config"},
            ],
        },
        asynchronous=asynchronous,
    )

    assert fields == {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {
                "role": "user",
                "content": [{"type": "text", "text": "again", "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
            },
        ],
        "system": [{"type": "text", "text": "Be terse.", "cache_control": {"type": "ephemeral"}}],
        "tools": [{**tool, "cache_control": {"type": "ephemeral"}}],
        "metadata": caller_metadata,
    }
    assert gateway == {"litellm_gateway_injected_cache": "deployment-1"}
    assert caller_metadata == {"user_id": "u"}


@pytest.mark.parametrize(
    ("asynchronous", "expected_system"),
    [
        (True, [{"type": "text", "text": "Be terse.", "cache_control": {"type": "ephemeral"}}]),
        (False, "Be terse."),
    ],
)
def test_body_params_prepare_prompt_caching_as_the_entry_that_ran(
    monkeypatch: pytest.MonkeyPatch, asynchronous: bool, expected_system: object
) -> None:
    monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", True)

    fields: Final = messages_host.body_params(
        {
            "model": "anthropic/claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "hi"}],
            "system": "Be terse.",
            "cache_control_injection_points": [{"location": "message", "role": "assistant"}],
        },
        asynchronous=asynchronous,
    )

    assert fields["system"] == expected_system


@pytest.mark.parametrize(
    ("modify_params", "args", "kwargs", "expected"),
    [
        pytest.param(True, (), {"max_tokens": 200_000}, 64_000, id="clamped-to-model-output-limit"),
        pytest.param(True, (), {"max_tokens": 1}, 1, id="under-the-limit"),
        pytest.param(False, (), {"max_tokens": 200_000}, 200_000, id="modify-params-off"),
        pytest.param("true", (), {"max_tokens": 200_000}, 200_000, id="truthy-but-not-true"),
        pytest.param(True, (200_000,), {"max_tokens": 200_000}, 200_000, id="positional-model-is-not-a-model"),
    ],
)
def test_modify_max_tokens_clamps_the_keyword_view_as_the_client_wrapper_does(
    monkeypatch: pytest.MonkeyPatch,
    modify_params: object,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    expected: int,
) -> None:
    monkeypatch.setattr(litellm, "modify_params", modify_params)
    model: Final = args[0] if args else "anthropic/claude-sonnet-4-5"
    view: Final = {**kwargs, "messages": [{"role": "user", "content": "hi"}]}

    messages_host.modify_max_tokens(model, args, view)

    assert view["max_tokens"] == expected


def test_workload_identity_token_mints_with_the_params_python_validates(monkeypatch: pytest.MonkeyPatch) -> None:
    minted: Final[list[tuple[dict[str, object], str | None, str]]] = []

    def mint(litellm_params: dict[str, object], api_base: str | None, model: str) -> str:
        federation = {name: value for name, value in litellm_params.items() if name.startswith("anthropic_")}
        minted.append(({name: value for name, value in federation.items() if value is not None}, api_base, model))
        return "sk-ant-oat01-minted"

    monkeypatch.setattr(messages_host, "get_anthropic_wif_token", mint)

    token: Final = messages_host.workload_identity_token(
        {"anthropic_federation_rule_id": "fdrl_1", "anthropic_issuer_ttl_seconds": "60"},
        "https://gateway.test",
    )

    assert (token, minted) == (
        "sk-ant-oat01-minted",
        [({"anthropic_federation_rule_id": "fdrl_1", "anthropic_issuer_ttl_seconds": 60}, "https://gateway.test", "")],
    )


def test_workload_identity_token_is_none_without_a_federation_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("ANTHROPIC_FEDERATION_RULE_ID", "ANTHROPIC_ORGANIZATION_ID", "ANTHROPIC_IDENTITY_TOKEN"):
        monkeypatch.delenv(name, raising=False)

    assert messages_host.workload_identity_token({}, None) is None


def test_workload_identity_fields_are_the_federation_litellm_params() -> None:
    assert set(messages_host.WORKLOAD_IDENTITY_FIELDS) >= {
        "anthropic_federation_rule_id",
        "anthropic_organization_id",
        "anthropic_disable_workload_identity_federation",
    }


class _Handler(AsyncHTTPHandler):
    def __init__(self, answer: httpx.Response | Exception) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  # the handler's untyped concurrent_limit
        self.answer: Final = answer
        self.posted: list[Mapping[str, object]] = []

    async def post(self, url: str, **kwargs: object) -> httpx.Response:  # pyright: ignore[reportIncompatibleMethodOverride]  # kwargs-ok: records the post's keywords
        self.posted.append({"url": url, **kwargs})
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


_URL: Final = "https://api.anthropic.com/v1/messages"
_REQUEST: Final = httpx.Request("POST", _URL)


@pytest.fixture
def no_tls_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("SSL_SECURITY_LEVEL", "SSL_ECDH_CURVE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(litellm, "ssl_security_level", None)
    monkeypatch.setattr(litellm, "ssl_ecdh_curve", None)


@pytest.mark.usefixtures("no_tls_settings")
@pytest.mark.parametrize(
    "client", (None, object(), httpx.AsyncClient(), HTTPHandler()), ids=("none", "object", "httpx", "sync-handler")
)
def test_the_native_client_stands_in_for_any_client_python_ignores(client: object) -> None:
    assert messages_host.transport_handler({"client": client}) == (None, [])


@pytest.mark.usefixtures("no_tls_settings")
def test_a_caller_async_handler_is_the_handler_python_posts_through() -> None:
    client: Final = AsyncHTTPHandler()

    assert messages_host.transport_handler({"client": client}) == (client, [])


@pytest.mark.usefixtures("no_tls_settings")
@pytest.mark.parametrize(
    "setting", ({"ssl_security_level": "DEFAULT@SECLEVEL=1"}, {"ssl_ecdh_curve": "secp521r1"}), ids=("level", "curve")
)
def test_tls_the_native_client_cannot_apply_posts_through_pythons_anthropic_handler(
    monkeypatch: pytest.MonkeyPatch, setting: Mapping[str, str]
) -> None:
    for name, value in setting.items():
        monkeypatch.setattr(litellm, name, value)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())

    handler, _ = messages_host.transport_handler({"client": object()})

    assert handler is get_async_httpx_client(llm_provider=litellm.LlmProviders.ANTHROPIC)
    assert callbacks_legacy_python.times_provider_exchange(object(), None) is False


@pytest.fixture
def fips_cipher_exclusion() -> Iterator[None]:
    exclude_rsa_key_exchange_and_chacha20(True)
    yield
    exclude_rsa_key_exchange_and_chacha20(False)


@pytest.mark.usefixtures("no_tls_settings", "fips_cipher_exclusion")
def test_fips_cipher_exclusion_posts_through_pythons_anthropic_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())

    handler, _ = messages_host.transport_handler({"client": None})

    assert handler is get_async_httpx_client(llm_provider=litellm.LlmProviders.ANTHROPIC)
    assert callbacks_legacy_python.times_provider_exchange(None, None) is False


@pytest.mark.usefixtures("no_tls_settings")
def test_only_a_native_post_is_timed_by_the_native_call() -> None:
    assert callbacks_legacy_python.times_provider_exchange(None, None) is True
    assert callbacks_legacy_python.times_provider_exchange(None, 30.0) is True
    assert callbacks_legacy_python.times_provider_exchange(AsyncHTTPHandler(), None) is False


@pytest.mark.usefixtures("no_tls_settings")
@pytest.mark.parametrize(
    "timeout",
    (-1.0, 0, -0.5, httpx.Timeout(10.0, connect=0)),
    ids=("negative", "zero", "fraction", "httpx-zero-connect"),
)
def test_a_non_positive_timeout_posts_through_pythons_anthropic_handler(
    monkeypatch: pytest.MonkeyPatch, timeout: object
) -> None:
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())

    handler, _ = messages_host.transport_handler({"client": None, "timeout": timeout})

    assert handler is get_async_httpx_client(llm_provider=litellm.LlmProviders.ANTHROPIC)
    assert callbacks_legacy_python.times_provider_exchange(None, timeout) is False


@pytest.mark.usefixtures("no_tls_settings")
@pytest.mark.parametrize("timeout", (None, 0.5, httpx.Timeout(10.0)), ids=("unset", "positive", "httpx"))
def test_a_positive_timeout_keeps_the_native_client(timeout: object) -> None:
    assert messages_host.transport_handler({"client": None, "timeout": timeout}) == (None, [])


@pytest.mark.usefixtures("no_tls_settings")
@pytest.mark.parametrize(
    "api_base",
    (
        "http://user:secret@host.example/d",
        "http://user@host.example/d",
        "http:///d",
        "http://:443/d",
        "http://host.example:99999/d",
        "http://host.example:port/d",
    ),
    ids=("credentials", "user", "no-host", "port-only", "port-out-of-range", "port-not-a-number"),
)
@pytest.mark.parametrize("model", ("anthropic/claude-x", "azure_ai/claude-x"))
def test_a_base_the_native_parser_reads_otherwise_posts_through_pythons_handler(
    monkeypatch: pytest.MonkeyPatch, api_base: str, model: str
) -> None:
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())

    handler, read = messages_host.transport_handler({"client": None, "model": model, "api_base": api_base})

    assert handler is get_async_httpx_client(llm_provider=litellm.LlmProviders.ANTHROPIC)
    assert read == []


@pytest.mark.usefixtures("no_tls_settings")
@pytest.mark.parametrize(
    "api_base",
    ("https://host.example/d", "http://host.example:8080/d?x=1", "https://[::1]:443/d"),
    ids=("plain", "port-and-query", "ipv6"),
)
@pytest.mark.parametrize("model", ("anthropic/claude-x", "azure_ai/claude-x"))
def test_a_plain_base_keeps_the_native_client(api_base: str, model: str) -> None:
    assert messages_host.transport_handler({"client": None, "model": model, "api_base": api_base}) == (None, [])


@pytest.mark.usefixtures("no_tls_settings")
@pytest.mark.parametrize(
    ("environment", "hosted"),
    ((" https://env.example/d ", True), ("https://env.example/d", False), (None, False)),
    ids=("padded", "plain", "unset"),
)
@pytest.mark.parametrize("api_base", (None, ""), ids=("absent", "empty"))
def test_azure_ai_reads_azure_api_base_once_and_hands_it_to_the_native_route(
    monkeypatch: pytest.MonkeyPatch, environment: str | None, hosted: bool, api_base: str | None
) -> None:
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    reads: Final[list[str]] = []

    def get_secret_str(name: str) -> str | None:
        reads.append(name)
        return environment

    monkeypatch.setattr(messages_host, "get_secret_str", get_secret_str)

    handler, read = messages_host.transport_handler(
        {"client": None, "model": "azure_ai/claude-x", "api_base": api_base}
    )

    assert (handler is not None) is hosted
    assert read == [("AZURE_API_BASE", environment)]
    assert reads == ["AZURE_API_BASE"]


@pytest.mark.usefixtures("no_tls_settings")
@pytest.mark.parametrize(
    ("environment", "hosted", "read"),
    (
        (
            {"ANTHROPIC_API_BASE": "http:///d/anthropic", "ANTHROPIC_BASE_URL": "https://b.example"},
            True,
            [("ANTHROPIC_API_BASE", "http:///d/anthropic")],
        ),
        (
            {"ANTHROPIC_BASE_URL": "http://user:pass@h.example/anthropic"},
            True,
            [("ANTHROPIC_API_BASE", None), ("ANTHROPIC_BASE_URL", "http://user:pass@h.example/anthropic")],
        ),
        (
            {"ANTHROPIC_API_BASE": "   ", "ANTHROPIC_BASE_URL": "https://b.example"},
            True,
            [("ANTHROPIC_API_BASE", "   ")],
        ),
        (
            {"ANTHROPIC_API_BASE": "", "ANTHROPIC_BASE_URL": "https://b.example"},
            False,
            [("ANTHROPIC_API_BASE", ""), ("ANTHROPIC_BASE_URL", "https://b.example")],
        ),
        ({}, False, [("ANTHROPIC_API_BASE", None), ("ANTHROPIC_BASE_URL", None)]),
    ),
    ids=("no-host-api-base", "credentials-base-url", "whitespace-api-base", "empty-api-base", "unset"),
)
@pytest.mark.parametrize("api_base", (None, ""), ids=("absent", "empty"))
def test_anthropic_env_base_is_guarded_like_a_caller_base_and_handed_to_the_native_route(
    monkeypatch: pytest.MonkeyPatch,
    environment: Mapping[str, str],
    hosted: bool,
    read: list[tuple[str, str | None]],
    api_base: str | None,
) -> None:
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    monkeypatch.setattr(messages_host, "get_secret_str", environment.get)

    handler, sent_read = messages_host.transport_handler(
        {"client": None, "model": "anthropic/claude-x", "api_base": api_base}
    )

    assert (handler is not None) is hosted
    assert sent_read == read


@pytest.mark.usefixtures("no_tls_settings")
@pytest.mark.parametrize(
    "request_",
    (
        {"client": None, "model": "anthropic/claude-x", "api_base": "https://call.example/d"},
        {"client": None, "model": "azure_ai/claude-x", "api_base": "https://call.example/d"},
        {"client": AsyncHTTPHandler(), "model": "azure_ai/claude-x"},
        {"client": None, "model": "azure_ai/claude-x", "timeout": 0},
    ),
    ids=("anthropic-call-base", "azure-call-base", "azure-caller-handler", "azure-python-timeout"),
)
def test_azure_api_base_is_left_to_the_native_route_when_the_choice_does_not_need_it(
    monkeypatch: pytest.MonkeyPatch, request_: Mapping[str, object]
) -> None:
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())

    def get_secret_str(name: str) -> str | None:
        raise AssertionError(name)

    monkeypatch.setattr(messages_host, "get_secret_str", get_secret_str)

    _, read = messages_host.transport_handler(request_)

    assert read == []


@pytest.mark.asyncio
async def test_transport_send_posts_the_native_wire_bytes_unchanged() -> None:
    logger: Final = object()
    wire: Final = '{"model":"claude-sonnet-4-5","messages":[{"role":"user","content":"h\u00e9llo"}]}'.encode()
    timeout: Final = httpx.Timeout(30.0, connect=2.0)
    handler: Final = _Handler(
        httpx.Response(200, json={"id": "msg_1"}, headers={"request-id": "req_1"}, request=_REQUEST)
    )

    sent: Final = await messages_host.transport_send(
        handler,
        {"litellm_logging_obj": logger, "timeout": timeout},
        _URL,
        [("x-api-key", "sk-ant"), ("anthropic-version", "2023-06-01")],
        wire,
        False,
    )

    assert handler.posted == [
        {
            "url": _URL,
            "headers": {"x-api-key": "sk-ant", "anthropic-version": "2023-06-01"},
            "data": wire,
            "stream": False,
            "logging_obj": logger,
            "timeout": timeout,
        }
    ]
    assert (sent.status, sent.text, sent.chunks, sent.rejection) == (200, '{"id":"msg_1"}', None, None)
    assert ("request-id", "req_1") in sent.headers


@pytest.mark.asyncio
async def test_transport_send_hands_signed_bytes_on_unchanged() -> None:
    handler: Final = _Handler(httpx.Response(200, json={}, request=_REQUEST))

    await messages_host.transport_send(handler, {}, _URL, [], b'{"signed":true}', False)

    assert handler.posted[0]["data"] == b'{"signed":true}'
    assert handler.posted[0]["timeout"] is None


@pytest.mark.asyncio
async def test_transport_send_leaves_a_stream_for_the_route_to_read() -> None:
    handler: Final = _Handler(
        httpx.Response(200, stream=httpx.ByteStream(b"event: message_stop\n\n"), request=_REQUEST)
    )

    sent: Final = await messages_host.transport_send(handler, {}, _URL, [], b"", True)

    assert sent.chunks is not None
    assert sent.text is None
    assert [chunk async for chunk in sent.chunks] == [b"event: message_stop\n\n"]


@pytest.mark.asyncio
@pytest.mark.parametrize("checked", (True, False), ids=("handler-raised", "status-checked"))
async def test_transport_send_reports_a_rejection_with_the_status_error(checked: bool) -> None:
    rejected: Final = httpx.Response(429, text="slow down", headers={"retry-after": "7"}, request=_REQUEST)
    error: Final = httpx.HTTPStatusError("rate limited", request=_REQUEST, response=rejected)
    handler: Final = _Handler(error if checked else rejected)

    sent: Final = await messages_host.transport_send(handler, {}, _URL, [], b"", False)

    assert (sent.status, sent.text) == (429, "slow down")
    assert ("retry-after", "7") in sent.headers
    assert isinstance(sent.rejection, httpx.HTTPStatusError)
    assert sent.rejection.response is rejected


@pytest.mark.asyncio
async def test_a_rejection_python_cannot_read_reaches_the_caller_as_raised() -> None:
    unread: Final = httpx.Response(500, stream=httpx.ByteStream(b"unread"), request=_REQUEST)

    with pytest.raises(messages_host.RawTransportFailure) as raised:
        await messages_host.transport_send(_Handler(unread), {}, _URL, [], b"", True)

    assert isinstance(raised.value.error, httpx.ResponseNotRead)
    assert isinstance(raised.value.error.__context__, httpx.HTTPStatusError)


@pytest.mark.asyncio
async def test_a_handler_failure_is_raised_for_the_route_to_map() -> None:
    failure: Final = httpx.ConnectError("refused")

    with pytest.raises(httpx.ConnectError) as raised:
        await messages_host.transport_send(_Handler(failure), {}, _URL, [], b"", False)

    assert raised.value is failure


@pytest.mark.parametrize(
    ("error", "expected", "status"),
    (
        pytest.param(
            litellm.Timeout(
                message="Read timed out.", model="default-model-name", llm_provider="litellm-httpx-handler"
            ),
            litellm.Timeout,
            408,
            id="timeout",
        ),
        pytest.param(
            httpx.ConnectError("All connection attempts failed"), litellm.InternalServerError, 500, id="connect"
        ),
        pytest.param(
            httpx.HTTPStatusError(
                "unavailable",
                request=_REQUEST,
                response=httpx.Response(503, text="Unavailable", request=_REQUEST),
            ),
            litellm.ServiceUnavailableError,
            503,
            id="status",
        ),
    ),
)
def test_a_handler_failure_maps_as_the_python_handler_maps_it(
    error: Exception, expected: type[Exception], status: int
) -> None:
    mapped: Final = messages_host.map_transport_failure(
        error, {"model": "anthropic/claude-sonnet-4-5", "api_base": None}, "anthropic"
    )

    assert isinstance(mapped, expected)
    assert getattr(mapped, "status_code", None) == status
    assert mapped.__context__ is not None
    assert mapped.__context__.__context__ is error


class _Hook(CustomLogger):
    rewrite: Mapping[str, object] = MappingProxyType({})
    seen: tuple[tuple[str, list[dict[str, object]], dict[str, object]], ...] = ()

    async def async_pre_request_hook(
        self, model: str, messages: list[dict[str, object]], kwargs: dict[str, object]
    ) -> dict[str, object] | None:
        self.seen = (*self.seen, (model, list(messages), dict(kwargs)))
        return {**kwargs, **self.rewrite}


def _hook(rewrite: Mapping[str, object]) -> _Hook:
    hook: Final = _Hook()
    hook.rewrite = rewrite
    return hook


def _hook_request(**overrides: object) -> dict[str, object]:
    return {
        "max_tokens": 64,
        "messages": [{"role": "user", "content": [{"type": "text", "text": ""}, {"type": "text", "text": "hi"}]}],
        "model": "anthropic/claude-sonnet-4-5",
        "metadata": None,
        "stop_sequences": None,
        "stream": False,
        "system": None,
        "temperature": None,
        "thinking": None,
        "tool_choice": None,
        "tools": None,
        "top_k": None,
        "top_p": None,
        "api_key": "sk-test",
        "api_base": None,
        "client": None,
        "custom_llm_provider": None,
        "litellm_call_id": "call-1",
        **overrides,
    }


def _instance_hook() -> CustomLogger:
    logger: Final = CustomLogger()
    logger.async_pre_request_hook = _hook({}).async_pre_request_hook  # pyright: ignore[reportAttributeAccessIssue]  # rebind-ok: an instance-level override
    return logger


@pytest.mark.parametrize(
    ("callbacks", "expected"),
    [
        pytest.param([_hook({})], True, id="override"),
        pytest.param([CustomLogger()], False, id="base-logger"),
        pytest.param([_instance_hook()], True, id="instance-override"),
        pytest.param(["not-a-logger"], False, id="not-a-logger"),
    ],
)
def test_runs_pre_request_hooks_needs_a_global_override(
    monkeypatch: pytest.MonkeyPatch, callbacks: list[object], expected: bool
) -> None:
    monkeypatch.setattr(litellm, "callbacks", callbacks)

    assert messages_host.runs_pre_request_hooks() is expected


@pytest.mark.asyncio
async def test_pre_request_hooks_hand_the_handler_what_the_python_entry_hands_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hook: Final = _hook({"temperature": 0.2, "tools": [{"name": "t"}], "extra_headers": {"x-hooked": "1"}})
    monkeypatch.setattr(litellm, "callbacks", [hook])

    hooked: Final = await messages_host.pre_request_hooks(_hook_request(temperature=0.9))

    assert hook.seen == (
        (
            "anthropic/claude-sonnet-4-5",
            [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            {
                "tools": None,
                "stream": False,
                "litellm_params": {"custom_llm_provider": "anthropic"},
                "tool_choice": None,
                "litellm_call_id": "call-1",
            },
        ),
    )
    assert hooked["temperature"] == 0.2
    assert hooked["tools"] == [{"name": "t"}]
    assert hooked["extra_headers"] == {"x-hooked": "1"}
    assert hooked["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert hooked["custom_llm_provider"] is None
    assert "litellm_params" not in hooked


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["model", "max_tokens", "messages", "api_key", "api_base", "custom_llm_provider"])
async def test_pre_request_hooks_naming_a_handler_argument_raise_the_python_type_error(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.setattr(litellm, "callbacks", [_hook({name: "hooked"})])

    with pytest.raises(TypeError, match=f"functools.partial\\(\\) got multiple values for keyword argument '{name}'"):
        await messages_host.pre_request_hooks(_hook_request())


@pytest.mark.asyncio
async def test_pre_request_hooks_refuse_a_route_to_another_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "callbacks", [_hook({"litellm_params": {"custom_llm_provider": "openai"}})])

    with pytest.raises(litellm.UnsupportedParamsError, match="routing the call to openai") as refused:
        await messages_host.pre_request_hooks(_hook_request())

    assert refused.value.llm_provider == "anthropic"


@pytest.mark.asyncio
async def test_pre_request_hooks_keep_the_callers_provider_over_the_hook(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "callbacks", [_hook({"litellm_params": {"custom_llm_provider": "openai"}})])

    hooked: Final = await messages_host.pre_request_hooks(_hook_request(custom_llm_provider="anthropic"))

    assert hooked["custom_llm_provider"] == "anthropic"


def test_body_params_hand_hooked_messages_on_as_already_sanitized() -> None:
    hooked: Final = [{"role": "user", "content": [{"type": "text", "text": ""}]}]

    fields: Final = messages_host.body_params(
        {"model": "anthropic/claude-sonnet-4-5", "messages": hooked}, asynchronous=True, hooked=True
    )

    assert fields == {"messages": hooked}


class _Agentic(CustomLogger):
    async def async_should_run_agentic_loop(self, *args: object, **kwargs: object) -> tuple[bool, dict[str, object]]:  # pyright: ignore[reportIncompatibleMethodOverride]  # test double only needs the override
        return False, {}


def _agentic_logger(callbacks: Sequence[CustomLogger]) -> Logging:
    logger: Final = Logging(
        model="anthropic/claude-sonnet-4-5",
        messages=[],
        stream=False,
        call_type="anthropic_messages",
        start_time=datetime.datetime.now(),
        litellm_call_id="agentic",
        function_id="agentic",
        dynamic_success_callbacks=list(callbacks),
    )
    return logger


@pytest.mark.parametrize(
    ("callbacks", "dynamic", "request_extra", "expected"),
    [
        pytest.param([], [], {}, False, id="no-hook"),
        pytest.param([_Agentic()], [], {}, True, id="global-hook"),
        pytest.param([], [_Agentic()], {}, True, id="call-hook"),
        pytest.param([], [], {"_websearch_interception_converted_stream": True}, True, id="converted-stream"),
    ],
)
def test_runs_agentic_hooks_gates_as_the_python_handler_does(
    monkeypatch: pytest.MonkeyPatch,
    callbacks: list[object],
    dynamic: Sequence[CustomLogger],
    request_extra: Mapping[str, object],
    expected: bool,
) -> None:
    monkeypatch.setattr(litellm, "callbacks", callbacks)
    request: Final = {"model": "anthropic/claude-sonnet-4-5", "litellm_logging_obj": _agentic_logger(dynamic)}

    assert messages_host.runs_agentic_hooks({**request, **request_extra}) is expected
    assert messages_host.runs_agentic_hooks({"model": "anthropic/claude-sonnet-4-5"}) is False


def test_record_agentic_loop_names_the_requested_model_for_follow_up_calls() -> None:
    logger: Final = _agentic_logger([])
    messages_host.record_agentic_loop(
        {
            "model": "anthropic/claude-sonnet-4-5",
            "litellm_logging_obj": logger,
            "_websearch_interception_converted_stream": True,
        },
        "anthropic",
    )

    details: Final = logger.model_call_details
    assert details["agentic_loop_params"] == {
        "model": "anthropic/claude-sonnet-4-5",
        "custom_llm_provider": "anthropic",
    }
    assert details["websearch_interception_converted_stream"] is True


def test_agentic_stream_tags_the_head_only_when_a_hook_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.rust_bridge.messages.entrypoints import MessagesStreamHead

    request: Final = {
        "model": "anthropic/claude-sonnet-4-5",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi", "provider_specific_fields": {"x": 1}}]}
        ],
        "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}],
        "api_key": "sk-test",
        "litellm_call_id": "agentic",
        "litellm_logging_obj": _agentic_logger([]),
    }
    sent: Final = {
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 16,
        "stream": True,
        "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}],
    }
    requested: Final = {"max_tokens": 16, "stream": True, "tools": request["tools"]}
    head: Final = MessagesStreamHead()

    monkeypatch.setattr(litellm, "callbacks", [])
    messages_host.agentic_stream(head, request, sent, "claude-sonnet-4-5", "anthropic", requested, hooked=False)
    assert head.agentic is None

    monkeypatch.setattr(litellm, "callbacks", [_Agentic()])
    messages_host.agentic_stream(head, request, sent, "claude-sonnet-4-5", "anthropic", requested, hooked=False)
    agentic: Final = head.agentic
    assert isinstance(agentic, functools.partial)
    bound: Final[Mapping[str, object]] = agentic.keywords
    call: Final = bound["call"]
    assert isinstance(call, messages_host._AgenticCall)  # pyright: ignore[reportPrivateUsage]  # the partial binds the handler call
    assert call.messages == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert call.params == {"max_tokens": 16, "stream": True, "tools": request["tools"]}
    assert call.params["tools"] is request["tools"]
    assert call.kwargs["api_key"] == "sk-test"
    assert "tools" not in call.kwargs


_HOISTED: Final = [{"type": "text", "text": "S"}, {"type": "text", "text": "R"}]


@pytest.mark.parametrize(
    ("sent_system", "requested_system", "expected"),
    [
        pytest.param({"system": _HOISTED}, {"system": "S"}, {"system": "S"}, id="caller-system-and-hoisted-turn"),
        pytest.param({"system": _HOISTED[1:]}, {}, {}, id="hoisted-turn-only"),
        pytest.param({}, {"system": "S"}, {}, id="system-not-sent"),
    ],
)
def test_agentic_params_take_the_callers_system_not_the_hoisted_wire_system(
    sent_system: Mapping[str, object], requested_system: Mapping[str, object], expected: Mapping[str, object]
) -> None:
    sent: Final = {"model": "claude-sonnet-4-5", "messages": [], "max_tokens": 16, "stream": False, **sent_system}
    requested: Final = {"max_tokens": 16, "stream": False, **requested_system}

    params: Final = messages_host._agentic_params(sent, requested, {"model": "azure_ai/claude-sonnet-4-5"})  # pyright: ignore[reportPrivateUsage]  # the follow-up params the handler's hooks read

    assert params == {"stream": False, "max_tokens": 16, **expected}


_DANGEROUS_TOOL_USE: Final = [{"type": "dangerous_tool_use"}]
_CLASSIFIER_SETTINGS: Final = {"safeguards_classifier_model": "openai/classifier"}
_NO_TOOL_USES: Final[dict[str, object]] = {}
_AVAILABLE: Final = ({"type": "dangerous_tool_use", "status": {"type": "available", "tool_uses": _NO_TOOL_USES}},)


@pytest.mark.parametrize(
    ("request_fields", "general_settings", "classifies"),
    (
        pytest.param(
            {"model": "anthropic/parity-tool-model", "safeguards": _DANGEROUS_TOOL_USE},
            _CLASSIFIER_SETTINGS,
            True,
            id="non-claude",
        ),
        pytest.param(
            {"model": "anthropic/claude-sonnet-4-5", "safeguards": _DANGEROUS_TOOL_USE},
            _CLASSIFIER_SETTINGS,
            False,
            id="claude",
        ),
        pytest.param(
            {"model": "anthropic/parity-tool-model", "safeguards": _DANGEROUS_TOOL_USE}, {}, False, id="unconfigured"
        ),
        pytest.param({"model": "anthropic/parity-tool-model"}, _CLASSIFIER_SETTINGS, False, id="not-requested"),
        pytest.param(
            {"model": "anthropic/parity-tool-model", "safeguards": [{"type": "other"}]},
            _CLASSIFIER_SETTINGS,
            False,
            id="other-safeguard",
        ),
    ),
)
def test_safeguards_classify_only_what_the_proxy_answers(
    monkeypatch: pytest.MonkeyPatch,
    request_fields: Mapping[str, object],
    general_settings: Mapping[str, object],
    classifies: bool,
) -> None:
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "general_settings", general_settings)

    assert (messages_host.safeguards(request_fields) is not None) is classifies


_OPUS_4_1_ON_BEDROCK: Final = "us.anthropic.claude-opus-4-1-20250805-v1:0"
_BEDROCK_PROFILE_ARN: Final = "arn:aws:bedrock:us-east-1:000000000000:application-inference-profile/abc123"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("deployment", "classifies"),
    (
        pytest.param(
            {"model": f"bedrock/invoke/{_OPUS_4_1_ON_BEDROCK}", "drop_params": True}, True, id="opus_4_1_drop_params"
        ),
        pytest.param(
            {"model": "bedrock/invoke/us.anthropic.claude-sonnet-4-5-20250929-v1:0", "drop_params": True},
            False,
            id="sonnet_4_5_drop_params",
        ),
        pytest.param(
            {"model": f"bedrock/invoke/{_OPUS_4_1_ON_BEDROCK}"}, True, id="opus_4_1_classifier_without_drop_params"
        ),
        pytest.param(
            {
                "model": f"bedrock/mantle/{_BEDROCK_PROFILE_ARN}",
                "base_model": f"bedrock/{_OPUS_4_1_ON_BEDROCK}",
                "drop_params": True,
            },
            True,
            id="profile_arn_of_opus_4_1",
        ),
    ),
)
async def test_safeguards_classify_exactly_what_the_bedrock_route_drops(
    local_model_cost_map: None,
    monkeypatch: pytest.MonkeyPatch,
    deployment: Mapping[str, object],
    classifies: bool,
) -> None:
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "general_settings", _CLASSIFIER_SETTINGS)
    monkeypatch.setattr(litellm, "drop_params", False)

    classifier: Final = messages_host.safeguards(
        {**deployment, "safeguards": _DANGEROUS_TOOL_USE, "messages": [{"role": "user", "content": "hi"}]}
    )

    assert (classifier is not None and await classifier.open_stream()) is classifies


@pytest.mark.parametrize(
    ("general_settings", "deployment", "drops"),
    (
        pytest.param(_CLASSIFIER_SETTINGS, {}, True, id="classifier_without_drop_params"),
        pytest.param({}, {"drop_params": True}, True, id="drop_params_without_classifier"),
        pytest.param({}, {}, False, id="neither"),
    ),
)
def test_settings_let_bedrock_drop_rejected_safeguards_under_drop_params_or_the_proxy_classifier(
    monkeypatch: pytest.MonkeyPatch,
    general_settings: Mapping[str, object],
    deployment: Mapping[str, object],
    drops: bool,
) -> None:
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "general_settings", general_settings)
    monkeypatch.setattr(litellm, "drop_params", False)

    projected: Final = messages_host.settings(
        deployment,
        {
            **_CALL,
            "model": f"bedrock/invoke/{_OPUS_4_1_ON_BEDROCK}",
            "safeguards": _DANGEROUS_TOOL_USE,
            **deployment,
            **_AWS,
        },
        asynchronous=True,
    )

    assert projected["drops_rejected_safeguards"] is drops


@pytest.mark.asyncio
async def test_safeguards_frames_split_and_stamp_like_the_python_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.llms.anthropic.messages.safeguards import SafeguardsEvaluator
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "general_settings", _CLASSIFIER_SETTINGS)
    evaluated: Final[list[object]] = []

    async def evaluate(_evaluator: SafeguardsEvaluator, tool_uses: object) -> object:
        evaluated.append(tool_uses)
        return _AVAILABLE

    monkeypatch.setattr(SafeguardsEvaluator, "evaluate", evaluate)
    classifier: Final = messages_host.safeguards(
        {
            "model": "anthropic/parity-tool-model",
            "safeguards": _DANGEROUS_TOOL_USE,
            "messages": [{"role": "user", "content": "hi"}],
        }
    )
    assert classifier is not None
    assert await classifier.open_stream()

    opened: Final = await classifier.frames(
        b'event: ping\ndata: {"type": "ping"}\n\nevent: message_delta\ndata: {"type": '
    )
    closed: Final = await classifier.frames(b'"message_delta", "delta": {"stop_reason": "end_turn"}}\n\ntail')
    flushed: Final = await classifier.flush()

    assert opened == (b'event: ping\ndata: {"type": "ping"}\n\n',)
    assert closed == (
        b'event: message_delta\ndata: {"type": "message_delta", "delta": {"stop_reason": "end_turn", '
        b'"safeguard_results": [{"type": "dangerous_tool_use", "status": {"type": "available", "tool_uses": {}}}]}}\n\n',
    )
    assert flushed == (b"tail",)
    assert evaluated == [()]


@pytest.mark.asyncio
async def test_safeguards_publish_the_results_python_stamped(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.llms.anthropic.messages.safeguards import SafeguardsEvaluator
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "general_settings", _CLASSIFIER_SETTINGS)
    stamped: Final = ({"type": "dangerous_tool_use", "status": {"type": "unsupported"}},)

    async def evaluate(_evaluator: SafeguardsEvaluator, tool_uses: object) -> object:
        return stamped

    monkeypatch.setattr(SafeguardsEvaluator, "evaluate", evaluate)
    request: Final = {
        "model": "anthropic/parity-tool-model",
        "safeguards": _DANGEROUS_TOOL_USE,
        "messages": [{"role": "user", "content": "hi"}],
    }
    unanswered: Final = messages_host.safeguards(request)
    answered: Final = messages_host.safeguards(request)
    assert unanswered is not None and answered is not None

    assert await unanswered.results({"content": [], "safeguard_results": []}) is None
    assert unanswered.published({"id": "msg"}) == {"id": "msg"}
    assert await answered.results({"content": []}) is stamped
    assert answered.published({"id": "msg", "safeguard_results": [stamped[0]]}) == {
        "id": "msg",
        "safeguard_results": stamped,
    }


class _Claims(MessagesInterceptor):
    def __init__(self, provider: str) -> None:
        self.provider = provider
        self.asked: list[tuple[object, str | None]] = []

    def can_handle(self, tools: Sequence[Mapping[str, object]] | None, custom_llm_provider: str | None) -> bool:
        self.asked.append((tools, custom_llm_provider))
        return custom_llm_provider == self.provider

    async def handle(self, **kwargs: object) -> object:  # pyright: ignore[reportIncompatibleMethodOverride]  # kwargs-ok: the interceptor contract's keyword call
        return kwargs


_TOOLS: Final = [{"type": "custom_tool", "name": "t"}]


@pytest.mark.parametrize(
    ("request_fields", "provider"),
    (
        pytest.param({"model": "anthropic/claude-sonnet-4-5"}, "anthropic", id="prefixed"),
        pytest.param({"model": "claude-sonnet-4-5", "custom_llm_provider": "anthropic"}, "anthropic", id="explicit"),
        pytest.param({"model": "not-a-known-model"}, None, id="unresolved"),
    ),
)
def test_interceptor_asks_with_the_tools_and_the_provider_the_handler_resolves(
    monkeypatch: pytest.MonkeyPatch, request_fields: Mapping[str, object], provider: str | None
) -> None:
    declined: Final = _Claims("other")
    claims: Final = _Claims("anthropic")
    monkeypatch.setattr(interceptors, "_interceptors", [declined, claims])

    chosen: Final = messages_host.interceptor({**request_fields, "tools": _TOOLS})

    assert chosen is (claims if provider == "anthropic" else None)
    assert declined.asked == [(_TOOLS, provider)]


@pytest.mark.asyncio
async def test_answer_hands_the_interceptor_the_call_as_the_handler_does() -> None:
    owner: Final = _Claims("anthropic")
    request: Final = {
        "model": "anthropic/claude-sonnet-4-5",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "text", "text": ""}]}],
        "max_tokens": 16,
        "tools": _TOOLS,
        "stream": None,
        "system": None,
        "client": object(),
        "custom_llm_provider": None,
        "litellm_call_id": "call-1",
    }

    handed: Final = await messages_host.answer(owner, request, stream=False, hooked=False)

    assert isinstance(handed, dict)
    assert handed["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert handed["custom_llm_provider"] == "anthropic"
    assert handed["stream"] is False
    assert handed["litellm_call_id"] == "call-1"
    assert "client" not in handed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stream", "converted", "expected"),
    (
        pytest.param(True, False, True, id="caller-stream"),
        pytest.param(False, True, True, id="converted-stream"),
        pytest.param(False, False, False, id="unary"),
    ),
)
async def test_answer_hands_a_hooked_call_on_with_the_callers_stream(
    stream: bool, converted: bool, expected: bool
) -> None:
    hooked_messages: Final = [{"role": "user", "content": [{"type": "text", "text": ""}]}]
    request: Final = {
        "model": "anthropic/claude-sonnet-4-5",
        "messages": hooked_messages,
        "max_tokens": 16,
        "tools": _TOOLS,
        "stream": False,
        "system": "hooked",
        "custom_llm_provider": None,
        **({"_websearch_interception_converted_stream": True} if converted else {}),
    }

    handed: Final = await messages_host.answer(_Claims("anthropic"), request, stream=stream, hooked=True)

    assert isinstance(handed, dict)
    assert handed["messages"] is hooked_messages
    assert handed["system"] == "hooked"
    assert handed["stream"] is expected


MCP_REFERENCE: Final = MappingProxyType(
    {"type": "mcp", "server_label": "litellm", "server_url": "litellm_proxy/mcp/deepwiki", "require_approval": "never"}
)


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        pytest.param({"tools": [dict(MCP_REFERENCE)]}, True, id="gateway-reference"),
        pytest.param({"tools": [{**MCP_REFERENCE, "server_url": "litellm_proxy"}]}, True, id="whole-gateway"),
        pytest.param({"tools": [dict(MCP_REFERENCE)], "_skip_mcp_handler": True}, False, id="gateway-follow-up"),
        pytest.param({"tools": [{"name": "a", "input_schema": {"type": "object"}}]}, False, id="native-tool"),
        pytest.param({"tools": []}, False, id="no-tools"),
        pytest.param({"tools": [dict(MCP_REFERENCE)], "mock_response": "hi"}, False, id="mock-answers-first"),
    ],
)
def test_settings_hand_the_call_to_the_mcp_gateway_where_the_python_handler_does(
    extra: dict[str, object], expected: bool
) -> None:
    settings: Final = messages_host.settings({}, {**_CALL, "model": "anthropic/claude-sonnet-4-5", **extra})

    assert settings["mcp_gateway"] is expected


def test_gateway_is_none_for_a_call_the_handler_serves_itself() -> None:
    assert messages_host.gateway({**_CALL, "model": "anthropic/claude-sonnet-4-5"}, True) is None


def _gateway_request() -> dict[str, object]:
    return {
        **_CALL,
        "model": "anthropic/claude-sonnet-4-5",
        "messages": [{"role": "user", "content": [{"type": "text", "text": ""}, {"type": "text", "text": "hi"}]}],
        "tools": [dict(MCP_REFERENCE)],
        "metadata": {"user_id": "u-1", "session": "dropped"},
        "stream": True,
        "temperature": 0.2,
        "litellm_metadata": {"tags": ["t1"]},
    }


def test_sync_gateway_hands_the_mcp_run_what_the_python_handler_hands_it(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.llms.anthropic.messages.native import handler

    sentinel: Final = object()
    handed: Final[list[dict[str, object]]] = []

    def run(**kwargs: object) -> object:
        handed.append(kwargs)
        return sentinel

    monkeypatch.setattr(mcp_handler, "anthropic_messages_with_mcp", run)
    request: Final = _gateway_request()

    assert handler.anthropic_messages_handler(**_gateway_request()) is sentinel  # noqa: TID251  # the Python handler under comparison
    assert messages_host.gateway(request, False) is sentinel
    assert handed[1] == handed[0]
    assert handed[1]["model"] == "claude-sonnet-4-5"
    assert handed[1]["custom_llm_provider"] == "anthropic"
    assert handed[1]["metadata"] == {"user_id": "u-1"}
    assert handed[1]["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert "_skip_mcp_handler" not in handed[1]


def test_async_gateway_tells_the_logger_the_agentic_loop_params_as_the_handler_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.litellm_core_utils.litellm_logging import Logging

    sentinel: Final = object()
    handed: Final[list[dict[str, object]]] = []

    def run(**kwargs: object) -> object:
        handed.append(kwargs)
        return sentinel

    monkeypatch.setattr(mcp_handler, "anthropic_messages_with_mcp", run)
    logger: Final = Logging(
        model="anthropic/claude-sonnet-4-5",
        messages=[],
        stream=False,
        call_type="aanthropic_messages",
        start_time=None,
        litellm_call_id="call-1",
        function_id="fn",
    )

    assert messages_host.gateway({**_gateway_request(), "litellm_logging_obj": logger}, True) is sentinel
    assert logger.model_call_details["agentic_loop_params"] == {
        "model": "anthropic/claude-sonnet-4-5",
        "custom_llm_provider": "anthropic",
    }
    assert logger.model_call_details["litellm_params"]["aanthropic_messages"] is True
    assert "is_async" not in handed[0]
    assert handed[0]["litellm_logging_obj"] is logger


def test_async_gateway_hands_hooked_messages_on_as_already_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    handed: Final[list[dict[str, object]]] = []

    def run(**kwargs: object) -> object:
        handed.append(kwargs)
        return None

    monkeypatch.setattr(mcp_handler, "anthropic_messages_with_mcp", run)
    hooked: Final = [
        {"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}]}
    ]

    messages_host.gateway({**_gateway_request(), "messages": hooked}, True, hooked=True)

    assert handed[0]["messages"] == hooked


def test_sync_answer_serves_the_gateway_after_a_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel: Final = object()

    def run(**_: object) -> object:
        return sentinel

    monkeypatch.setattr(mcp_handler, "anthropic_messages_with_mcp", run)

    served: Final = messages_host.sync_answer(_gateway_request())
    mocked: Final = messages_host.sync_answer({**_gateway_request(), "mock_response": "hi"})

    assert served is not None and served.value is sentinel
    assert mocked is not None and isinstance(mocked.value, dict)
    assert messages_host.sync_answer({**_CALL, "model": "anthropic/claude-sonnet-4-5"}) is None


@pytest.mark.parametrize(
    ("extra", "raised", "text"),
    [
        pytest.param({"messages": None}, TypeError, "'NoneType' object is not iterable", id="messages-none"),
        pytest.param({"messages": 5}, TypeError, "'int' object is not iterable", id="messages-int"),
        pytest.param({"tools": {"name": "t"}}, AttributeError, "'str' object has no attribute 'get'", id="tools-map"),
        pytest.param({"model_info": "not-a-mapping"}, ValidationError, "model_info", id="model-info-str"),
        pytest.param({"custom_llm_provider": 5}, ValidationError, "custom_llm_provider", id="provider-int"),
    ],
)
def test_settings_raise_what_the_python_handler_raises_for_arguments_of_the_wrong_type(
    extra: dict[str, object], raised: type[Exception], text: str
) -> None:
    with pytest.raises(raised, match=text):
        messages_host.settings({}, {**_CALL, "model": "anthropic/claude-sonnet-4-5", **extra})


def test_settings_leave_well_typed_tools_to_the_interceptors_after_the_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: Final[list[object]] = []

    def can_handle(tools: object, custom_llm_provider: object) -> bool:
        asked.append(tools)
        return False

    advisor: Final = interceptors.get_messages_interceptors()[0]
    monkeypatch.setattr(advisor, "can_handle", can_handle)

    messages_host.settings({}, {**_CALL, "model": "anthropic/claude-sonnet-4-5", "tools": [{"name": "t"}]})

    assert asked == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("hook_params", "raised", "text"),
    [
        pytest.param("not-a-mapping", AttributeError, "'str' object has no attribute 'get'", id="params-str"),
        pytest.param({"custom_llm_provider": 5}, ValidationError, "custom_llm_provider", id="provider-int"),
    ],
)
async def test_pre_request_hooks_raise_what_the_python_handler_raises_for_the_hooks_params(
    monkeypatch: pytest.MonkeyPatch, hook_params: object, raised: type[Exception], text: str
) -> None:
    monkeypatch.setattr(litellm, "callbacks", [_hook({"litellm_params": hook_params})])

    with pytest.raises(raised, match=text):
        await messages_host.pre_request_hooks(_hook_request(model="claude-sonnet-4-5"))


def test_a_provider_config_rejection_maps_as_the_python_http_handler_maps_it() -> None:
    logger: Final = Logging(
        model="anthropic/claude-sonnet-4-5",
        messages=[],
        stream=False,
        call_type="anthropic_messages",
        start_time=datetime.datetime.now(),
        litellm_call_id="rejected",
        function_id="rejected",
    )
    rejected: Final = ValueError("missing required field: max_tokens")
    rejected.messages_request_error = True  # pyright: ignore[reportAttributeAccessIssue]  # marker the native host sets
    rejected.messages_provider_rejection = "max_tokens is required for Anthropic /v1/messages API"  # pyright: ignore[reportAttributeAccessIssue]  # marker the native host sets

    mapped: Final = messages_host.map_failure(
        rejected,
        {"model": "anthropic/claude-sonnet-4-5", "messages": [], "litellm_logging_obj": logger},
        "anthropic",
    )

    assert isinstance(mapped, litellm.BadRequestError)
    assert str(mapped) == (
        "litellm.BadRequestError: AnthropicException - max_tokens is required for Anthropic /v1/messages API"
    )
    assert mapped.model == "anthropic/claude-sonnet-4-5"  # pyright: ignore[reportUnknownMemberType]  # the public error leaves model untyped
    assert mapped.llm_provider == "anthropic"  # pyright: ignore[reportUnknownMemberType]  # the public error leaves llm_provider untyped
    assert logger.model_call_details["custom_llm_provider"] == "anthropic"


def test_a_bad_request_another_config_raises_maps_by_its_text_as_the_python_http_handler_maps_it() -> None:
    text: Final = "Unmapped reasoning effort: 'ultra'. Must be one of: 'low', 'none'."
    rejected: Final = ValueError(text)
    rejected.messages_request_error = True  # pyright: ignore[reportAttributeAccessIssue]  # marker the native host sets
    rejected.messages_provider_rejection = text  # pyright: ignore[reportAttributeAccessIssue]  # marker the native host sets
    rejected.messages_provider_bad_request = True  # pyright: ignore[reportAttributeAccessIssue]  # marker the native host sets

    mapped: Final = messages_host.map_failure(
        rejected, {"model": "anthropic/claude-sonnet-4-5", "messages": []}, "anthropic"
    )

    assert type(mapped) is litellm.BadRequestError
    assert str(mapped) == f"litellm.BadRequestError: AnthropicException - litellm.BadRequestError: {text}"


def test_short_circuits_needs_a_replaced_web_search_short_circuit(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.integrations.websearch_interception.handler import WebSearchInterceptionLogger

    class _ShortCircuit(WebSearchInterceptionLogger):
        async def try_short_circuit_search(self, *args: object, **kwargs: object) -> dict[str, object] | None:  # pyright: ignore[reportIncompatibleMethodOverride]  # test double only needs the override
            return None

    monkeypatch.setattr(litellm, "callbacks", [CustomLogger()])
    assert not messages_host.short_circuits()
    monkeypatch.setattr(litellm, "callbacks", [WebSearchInterceptionLogger()])
    assert not messages_host.short_circuits()
    monkeypatch.setattr(litellm, "callbacks", [_ShortCircuit()])
    assert messages_host.short_circuits()


_INVALID_CLIENT: Final = json.dumps(
    {
        "error": "invalid_client",
        "error_description": "AADSTS7000215: Invalid client secret provided. Ensure the secret for app "
        "'00000000-0000-0000-0000-0000000000b1' is the value. Trace ID: 0b1f0000-0000-0000-0000-000000000001",
    }
).encode()
_UNKNOWN_APPLICATION: Final = json.dumps(
    {
        "error": "unauthorized_client",
        "error_description": "AADSTS700016: Application with identifier '11111111-2222-3333-4444-555555555555' "
        "was not found in the directory 'tenant'. Trace ID: x",
    }
).encode()
_TRANSIENT: Final = json.dumps({"error": "temporarily_unavailable", "error_description": "AADSTS90033: Try again."})
_JSON: Final = (("content-type", "application/json"),)


# The "Authentication failed: ..." texts are azure-identity 1.25.2 and MSAL 1.37.0 wording, checked 2026-11-13:
# MSAL raises "HTTP Error: <status>" for a 5xx (throttled_http_client.py), azure-core 1.41.0 raises "Invalid HTTP
# date" for an unparseable Retry-After (pipeline/policies/_utils.py), CPython 3.12.3's time.sleep refuses a wait
# past its time_t range and json raises its own decode error.
@pytest.mark.parametrize(
    ("status", "headers", "body", "raised", "status_code", "message"),
    [
        pytest.param(
            401,
            _JSON,
            _INVALID_CLIENT,
            EntraRejectionError,
            401,
            "Failed to get Azure AD token: AADSTS7000215: Invalid client secret provided.",
            id="listed_rejection_is_a_401",
        ),
        pytest.param(
            400,
            _JSON,
            _UNKNOWN_APPLICATION,
            EntraRejectionError,
            401,
            "Failed to get Azure AD token: AADSTS700016: Application with identifier '<redacted>' was not found "
            "in the directory 'tenant'.",
            id="rejection_redacts_guids",
        ),
        pytest.param(
            500,
            _JSON,
            _TRANSIENT.encode(),
            RuntimeError,
            None,
            "Failed to get Azure AD token: Authentication failed: HTTP Error: 500",
            id="exhausted_server_error",
        ),
        pytest.param(
            500,
            (*_JSON, ("retry-after", "soon")),
            _TRANSIENT.encode(),
            RuntimeError,
            None,
            "Failed to get Azure AD token: Authentication failed: Invalid HTTP date",
            id="malformed_retry_after",
        ),
        pytest.param(
            503,
            (*_JSON, ("retry-after", "inf")),
            _TRANSIENT.encode(),
            RuntimeError,
            None,
            "Failed to get Azure AD token: Authentication failed: timestamp out of range for platform time_t",
            id="retry_after_too_long_to_sleep",
        ),
        pytest.param(
            400,
            (("content-type", "text/html"),),
            b"<html>nope</html>",
            RuntimeError,
            None,
            "Failed to get Azure AD token: Authentication failed: Expecting value: line 1 column 1 (char 0)",
            id="answer_that_is_not_json",
        ),
    ],
)
def test_a_native_token_endpoint_answer_raises_what_get_azure_ad_token_raises(
    status: int,
    headers: tuple[tuple[str, str], ...],
    body: bytes,
    raised: type[Exception],
    status_code: int | None,
    message: str,
) -> None:
    failure: Final = messages_host.entra_token_response_failure(status, headers, body)

    assert type(failure) is raised
    assert str(failure) == message
    assert getattr(failure, "status_code", None) == status_code
    assert failure.__cause__ is not None


def test_a_failing_token_provider_logs_and_raises_what_get_azure_ad_token_does(
    caplog: pytest.LogCaptureFixture,
) -> None:
    error: Final = ValueError("token service unavailable")

    with caplog.at_level(logging.ERROR, logger="LiteLLM"):
        failure: Final = messages_host.azure_ad_token_failure(error)

    assert type(failure) is RuntimeError
    assert str(failure) == "Failed to get Azure AD token: token service unavailable"
    assert failure.__cause__ is error
    assert caplog.messages == ["Error calling Azure AD token provider: token service unavailable"]


def test_a_non_string_token_logs_and_raises_what_get_azure_ad_token_does(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.ERROR, logger="LiteLLM"):
        failure: Final = messages_host.non_string_azure_ad_token(1)

    assert type(failure) is TypeError
    assert str(failure) == "Azure AD token must be a string, got <class 'int'>"
    assert caplog.messages == ["Azure AD token provider returned non-string value: <class 'int'>"]


_AZURE_CLAUDE_REQUEST: Final = MappingProxyType({"model": "azure_ai/claude-sonnet-4-5"})


@pytest.mark.parametrize(
    ("error", "raised", "status_code"),
    [
        pytest.param(
            EntraRejectionError(status_code=401, message="Failed to get Azure AD token: AADSTS50126: Bad password."),
            litellm.AuthenticationError,
            401,
            id="entra_rejection_is_an_authentication_error",
        ),
        pytest.param(
            AzureOpenAIError(status_code=422, message="Azure AD Token expires_in not returned"),
            litellm.BadRequestError,
            422,
            id="oidc_reply_without_fields_is_a_bad_request",
        ),
    ],
)
def test_a_litellm_token_failure_is_mapped_as_the_handler_maps_it(
    error: Exception, raised: type[Exception], status_code: int
) -> None:
    failure: Final = messages_host.map_token_failure(error, _AZURE_CLAUDE_REQUEST, "azure_ai")

    assert type(failure) is raised
    assert getattr(failure, "status_code", None) == status_code
    assert getattr(failure, "llm_provider", None) == "azure_ai"
    assert str(error) in str(failure)
    assert failure.__context__ is error


def test_any_other_token_failure_reaches_the_caller_as_raised() -> None:
    error: Final = RuntimeError("Failed to get Azure AD token: Authentication failed: HTTP Error: 500")

    assert messages_host.map_token_failure(error, _AZURE_CLAUDE_REQUEST, "azure_ai") is error


_PARAMS: Final = TypeAdapter(dict[str, object])


@pytest.mark.parametrize(
    ("provider", "asynchronous", "flagged"),
    [
        pytest.param("azure_ai", True, True, id="azure_ai_async"),
        pytest.param("azure_ai", False, False, id="azure_ai_sync"),
        pytest.param("anthropic", True, True, id="anthropic_async"),
        pytest.param("anthropic", False, False, id="anthropic_sync"),
        pytest.param("bedrock", True, True, id="bedrock_async"),
    ],
)
def test_a_token_failure_logs_the_call_flag_python_s_handler_set_first(
    provider: str, asynchronous: bool, flagged: bool
) -> None:
    logger: Final = Logging(
        model=f"{provider}/claude-sonnet-4-5",
        messages=[],
        stream=False,
        call_type="anthropic_messages",
        start_time=datetime.datetime.now(),
        litellm_call_id="token",
        function_id="token",
    )

    messages_host.map_token_failure(
        RuntimeError("Failed to get Azure AD token"),
        {"model": f"{provider}/claude-sonnet-4-5", "litellm_logging_obj": logger},
        provider,
        asynchronous,
    )

    details: Final[Mapping[str, object]] = logger.model_call_details
    params: Final = _PARAMS.validate_python(details.get("litellm_params", {}))
    assert params.get("aanthropic_messages") is flagged


@pytest.mark.parametrize(
    ("provider", "asynchronous", "flagged"),
    [
        pytest.param("azure_ai", True, True, id="azure_ai_async"),
        pytest.param("azure_ai", False, False, id="azure_ai_sync"),
        pytest.param("anthropic", True, True, id="anthropic_async"),
        pytest.param("anthropic", False, False, id="anthropic_sync"),
        pytest.param("bedrock", True, True, id="bedrock_async"),
    ],
)
def test_an_unsupported_param_logs_the_call_flag_python_s_handler_set_first(
    provider: str, asynchronous: bool, flagged: bool
) -> None:
    logger: Final = Logging(
        model=f"{provider}/claude-sonnet-4-5",
        messages=[],
        stream=False,
        call_type="anthropic_messages",
        start_time=datetime.datetime.now(),
        litellm_call_id="unsupported",
        function_id="unsupported",
    )
    rejected: Final = ValueError("temperature is not supported")
    rejected.messages_request_error = True  # pyright: ignore[reportAttributeAccessIssue]  # marker the native host sets
    rejected.messages_unsupported_param = True  # pyright: ignore[reportAttributeAccessIssue]  # marker the native host sets

    mapped: Final = messages_host.map_failure(
        rejected,
        {"model": f"{provider}/claude-sonnet-4-5", "messages": [], "litellm_logging_obj": logger},
        provider,
        asynchronous,
    )

    assert isinstance(mapped, litellm.UnsupportedParamsError)
    details: Final[Mapping[str, object]] = logger.model_call_details
    params: Final = _PARAMS.validate_python(details.get("litellm_params", {}))
    assert params.get("aanthropic_messages") is flagged


@pytest.mark.parametrize(
    ("attributed_model", "llm_provider", "model"),
    [
        pytest.param(
            "claude-sonnet-4-5", "vertex_ai", "claude-sonnet-4-5", id="vertex_version_names_provider_and_model"
        ),
        pytest.param(None, None, None, id="unattributed_names_neither"),
    ],
)
def test_an_attributed_unsupported_param_names_the_provider_and_model(
    attributed_model: str | None, llm_provider: str | None, model: str | None
) -> None:
    text: Final = (
        "claude-sonnet-4-5 does not support anthropic-version=latest. "
        "To drop unsupported params, set `litellm.drop_params = True`."
    )
    rejected: Final = ValueError(text)
    rejected.messages_request_error = True  # pyright: ignore[reportAttributeAccessIssue]  # marker the Rust host sets
    rejected.messages_unsupported_param = True  # pyright: ignore[reportAttributeAccessIssue]  # marker the Rust host sets
    if attributed_model is not None:
        rejected.messages_unsupported_param_model = attributed_model  # pyright: ignore[reportAttributeAccessIssue]  # marker the Rust host sets

    mapped: Final = messages_host.map_failure(
        rejected, {"model": "vertex_ai/claude-sonnet-4-5", "messages": []}, "vertex_ai", True
    )

    assert isinstance(mapped, litellm.UnsupportedParamsError)
    assert (mapped.llm_provider, mapped.model, mapped.status_code, str(mapped)) == (
        llm_provider,
        model,
        400,
        f"litellm.UnsupportedParamsError: {text}",
    )


@dataclass(frozen=True, slots=True)
class _TokenEndpointCertificates:
    ca: str
    valid: tuple[str, str]
    expired: tuple[str, str]


@pytest.fixture(scope="module")
def token_endpoint_certificates(tmp_path_factory: pytest.TempPathFactory) -> _TokenEndpointCertificates:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    directory: Final = tmp_path_factory.mktemp("entra")
    now: Final = datetime.datetime.now(datetime.timezone.utc)
    ca_key: Final = ec.generate_private_key(ec.SECP256R1())
    ca_name: Final = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "entra test CA")])
    ca: Final = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(ca_key, hashes.SHA256())
    )

    def leaf(name: str, not_after: datetime.datetime) -> tuple[str, str]:
        key: Final = ec.generate_private_key(ec.SECP256R1())
        certificate: Final = (
            x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "login.microsoftonline.com")]))
            .issuer_name(ca_name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(not_after - datetime.timedelta(days=30))
            .not_valid_after(not_after)
            .add_extension(x509.SubjectAlternativeName([x509.DNSName("login.microsoftonline.com")]), critical=False)
            .sign(ca_key, hashes.SHA256())
        )
        (directory / f"{name}.pem").write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
        (directory / f"{name}.key").write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
            )
        )
        return str(directory / f"{name}.pem"), str(directory / f"{name}.key")

    (directory / "ca.pem").write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    return _TokenEndpointCertificates(
        ca=str(directory / "ca.pem"),
        valid=leaf("valid", now + datetime.timedelta(days=1)),
        expired=leaf("expired", now - datetime.timedelta(days=1)),
    )


_ADDRESS: Final = TypeAdapter(tuple[str, int])


def _serve(handle: Callable[[socket.socket], None]) -> int:
    listener: Final = socket.create_server(("127.0.0.1", 0))

    def accept() -> None:
        while True:
            connection = listener.accept()[0]
            threading.Thread(target=handle, args=(connection,), daemon=True).start()

    threading.Thread(target=accept, daemon=True).start()
    return _port(listener)


def _reset(connection: socket.socket) -> None:
    connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    connection.close()


def _read_head(connection: socket.socket) -> None:
    received: Final = bytearray()
    while b"\r\n\r\n" not in received and (chunk := connection.recv(4096)):
        received.extend(chunk)


def _proxy(answer: Callable[[socket.socket], None]) -> int:
    def handle(connection: socket.socket) -> None:
        _read_head(connection)
        answer(connection)

    return _serve(handle)


def _tunnel_to(port: int) -> Callable[[socket.socket], None]:
    def tunnel(connection: socket.socket) -> None:
        upstream: Final = socket.create_connection(("127.0.0.1", port))
        connection.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")

        def pipe(source: socket.socket, target: socket.socket) -> None:
            with contextlib.suppress(OSError):
                while data := source.recv(65536):
                    target.sendall(data)
            with contextlib.suppress(OSError):
                target.shutdown(socket.SHUT_WR)

        threading.Thread(target=pipe, args=(upstream, connection), daemon=True).start()
        pipe(connection, upstream)

    return tunnel


def _tls_endpoint(certificate: tuple[str, str]) -> int:
    context: Final = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(*certificate)

    def handle(connection: socket.socket) -> None:
        with contextlib.suppress(OSError), context.wrap_socket(connection, server_side=True) as tls:
            _read_head(tls)

    return _serve(handle)


def _closed_port() -> int:
    with socket.create_server(("127.0.0.1", 0)) as listener:
        return _port(listener)


def _port(listener: socket.socket) -> int:
    return _ADDRESS.validate_python(listener.getsockname())[1]


def _ask_for_proxy_credentials(connection: socket.socket) -> None:
    connection.sendall(b"HTTP/1.1 407 Proxy Authentication Required\r\ncontent-length: 0\r\n\r\n")


def _close(connection: socket.socket) -> None:
    connection.close()


def _azure_identity_reason(monkeypatch: pytest.MonkeyPatch, environment: Mapping[str, str]) -> str:
    from azure.core.exceptions import ClientAuthenticationError
    from azure.identity import ClientSecretCredential

    for name in [name for name in os.environ if name.lower().endswith("_proxy") or name.endswith("CA_BUNDLE")]:
        monkeypatch.delenv(name)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    credential: Final = ClientSecretCredential(
        "tenant", "client", "secret", disable_instance_discovery="AZURE_AUTHORITY_HOST" in environment
    )
    with pytest.raises(ClientAuthenticationError) as raised:
        credential.get_token("https://cognitiveservices.azure.com/.default")
    return str(raised.value).removeprefix("Authentication failed: ")


def test_an_unreachable_proxy_is_worded_as_requests_words_it(monkeypatch: pytest.MonkeyPatch) -> None:
    port: Final = _closed_port()

    reason: Final = _azure_identity_reason(monkeypatch, {"HTTPS_PROXY": f"http://127.0.0.1:{port}"})

    assert reason == messages_host.requests_proxy_unreachable("127.0.0.1", port, errno.ECONNREFUSED)


@pytest.mark.parametrize(
    ("answer", "worded"),
    [
        pytest.param(
            _ask_for_proxy_credentials,
            messages_host.requests_proxy_authentication_required,
            id="proxy_asks_for_credentials",
        ),
        pytest.param(
            _close,
            lambda: messages_host.requests_connection_aborted(None),
            id="proxy_closes",
        ),
        pytest.param(_reset, lambda: messages_host.requests_connection_aborted(errno.ECONNRESET), id="proxy_resets"),
    ],
)
def test_a_proxy_refusing_the_tunnel_is_worded_as_requests_words_it(
    monkeypatch: pytest.MonkeyPatch, answer: Callable[[socket.socket], None], worded: Callable[[], str]
) -> None:
    reason: Final = _azure_identity_reason(monkeypatch, {"HTTPS_PROXY": f"http://127.0.0.1:{_proxy(answer)}"})

    assert reason == worded()


def test_a_connection_closed_in_the_handshake_is_worded_as_ssl_words_it(monkeypatch: pytest.MonkeyPatch) -> None:
    def close_before_handshake(connection: socket.socket) -> None:
        connection.shutdown(socket.SHUT_WR)
        connection.recv(65536)
        connection.close()

    reason: Final = _azure_identity_reason(
        monkeypatch, {"AZURE_AUTHORITY_HOST": f"https://127.0.0.1:{_serve(close_before_handshake)}"}
    )

    assert reason == messages_host.requests_tls_eof()


def test_a_tunnel_closed_before_the_answer_is_worded_as_requests_words_it(
    monkeypatch: pytest.MonkeyPatch, token_endpoint_certificates: _TokenEndpointCertificates
) -> None:
    endpoint: Final = _tls_endpoint(token_endpoint_certificates.valid)

    reason: Final = _azure_identity_reason(
        monkeypatch,
        {
            "HTTPS_PROXY": f"http://127.0.0.1:{_proxy(_tunnel_to(endpoint))}",
            "REQUESTS_CA_BUNDLE": token_endpoint_certificates.ca,
        },
    )

    assert reason == messages_host.requests_proxy_disconnected()


def test_an_expired_certificate_is_worded_as_openssl_words_it(
    monkeypatch: pytest.MonkeyPatch, token_endpoint_certificates: _TokenEndpointCertificates
) -> None:
    endpoint: Final = _tls_endpoint(token_endpoint_certificates.expired)

    reason: Final = _azure_identity_reason(
        monkeypatch,
        {
            "HTTPS_PROXY": f"http://127.0.0.1:{_proxy(_tunnel_to(endpoint))}",
            "REQUESTS_CA_BUNDLE": token_endpoint_certificates.ca,
        },
    )

    assert reason == messages_host.requests_certificate_rejected("certificate has expired")


def test_a_certificate_for_another_name_is_worded_as_ssl_words_it(
    monkeypatch: pytest.MonkeyPatch, token_endpoint_certificates: _TokenEndpointCertificates
) -> None:
    endpoint: Final = _tls_endpoint(token_endpoint_certificates.valid)

    reason: Final = _azure_identity_reason(
        monkeypatch,
        {"AZURE_AUTHORITY_HOST": f"https://localhost:{endpoint}", "REQUESTS_CA_BUNDLE": token_endpoint_certificates.ca},
    )

    assert reason == messages_host.requests_certificate_rejected(
        "Hostname mismatch, certificate is not valid for 'localhost'."
    )


def test_a_failure_without_an_endpoint_answer_raises_what_get_azure_ad_token_raises() -> None:
    failure: Final = messages_host.entra_token_failure("connection refused")

    assert type(failure) is RuntimeError
    assert str(failure) == "Failed to get Azure AD token: Authentication failed: connection refused"


def test_an_unresolvable_proxy_is_worded_as_requests_words_it(monkeypatch: pytest.MonkeyPatch) -> None:
    lookup: Final = "Name or service not known"
    resolve: Final = socket.getaddrinfo

    def unresolved(host: str, *args: object, **kwargs: object) -> object:  # kwargs-ok: getaddrinfo's own arguments
        if host == "proxy.invalid":
            raise socket.gaierror(socket.EAI_NONAME, lookup)
        return resolve(host, *args, **kwargs)  # pyright: ignore[reportArgumentType]  # getaddrinfo's own arguments, passed on

    monkeypatch.setattr(socket, "getaddrinfo", unresolved)

    reason: Final = _azure_identity_reason(monkeypatch, {"HTTPS_PROXY": "http://proxy.invalid:3128"})

    assert reason == messages_host.requests_proxy_unresolved("proxy.invalid", 3128, socket.EAI_NONAME, lookup)


def _hold(connection: socket.socket) -> None:
    _read_head(connection)
    with contextlib.suppress(OSError):
        connection.recv(1)


def test_a_token_endpoint_that_never_answers_is_worded_as_requests_words_it(
    monkeypatch: pytest.MonkeyPatch, token_endpoint_certificates: _TokenEndpointCertificates
) -> None:
    from azure.core.configuration import ConnectionConfiguration

    defaults: Final = ConnectionConfiguration.__init__.__kwdefaults__
    assert defaults is not None
    monkeypatch.setitem(defaults, "read_timeout", 0.05)
    context: Final = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(*token_endpoint_certificates.valid)

    def handle(connection: socket.socket) -> None:
        with contextlib.suppress(OSError), context.wrap_socket(connection, server_side=True) as tls:
            _hold(tls)

    endpoint: Final = _serve(handle)

    reason: Final = _azure_identity_reason(
        monkeypatch,
        {
            "HTTPS_PROXY": f"http://127.0.0.1:{_proxy(_tunnel_to(endpoint))}",
            "REQUESTS_CA_BUNDLE": token_endpoint_certificates.ca,
        },
    )

    assert reason == messages_host.requests_read_timeout("login.microsoftonline.com", 443)


def test_a_token_endpoint_answer_that_grants_the_token_raises_nothing() -> None:
    granted: Final = json.dumps({"access_token": "token", "token_type": "Bearer", "expires_in": 3600}).encode()

    assert messages_host.entra_token_response_failure(200, _JSON, granted) is None


@pytest.mark.usefixtures("no_tls_settings")
@pytest.mark.parametrize("model", ("anthropic/claude-x", "azure_ai/claude-x"))
def test_a_base_python_cannot_split_keeps_the_native_client(model: str) -> None:
    assert messages_host.transport_handler({"client": None, "model": model, "api_base": "https://[::1/d"}) == (
        None,
        [],
    )


@pytest.mark.parametrize(
    ("status", "body", "error_class", "text"),
    [
        (
            429,
            '{"message": "Too many requests"}',
            litellm.RateLimitError,
            'litellm.RateLimitError: BedrockException - {"message": "Too many requests"}',
        ),
        (
            400,
            '{"message": "prompt is too long: 300000 tokens > 200000 maximum"}',
            litellm.ContextWindowExceededError,
            "litellm.ContextWindowExceededError: litellm.BadRequestError: BedrockException: Context Window Error - "
            '{"message": "prompt is too long: 300000 tokens > 200000 maximum"}',
        ),
        (
            404,
            '{"message": "model not found"}',
            litellm.NotFoundError,
            'litellm.NotFoundError: BedrockException - {"message": "model not found"}',
        ),
    ],
    ids=("throttled", "context-window", "not-found"),
)
def test_a_bedrock_upstream_status_maps_to_the_bedrock_exception_python_raises(
    status: int, body: str, error_class: type[Exception], text: str
) -> None:
    request: Final[dict[str, object]] = {"model": _BEDROCK_INVOKE, "messages": [], **_AWS}
    upstream: Final = ValueError(status, body)
    upstream.headers = [("content-type", "application/json"), ("x-amzn-requestid", "req-1")]  # pyright: ignore[reportAttributeAccessIssue]  # set by the native host

    mapped: Final = messages_host.map_failure(upstream, request, messages_host.failure_provider(request))

    assert type(mapped) is error_class
    assert str(mapped) == text
    assert mapped.status_code == status  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # the public error's status
    assert mapped.llm_provider == "bedrock"  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # the public error's provider
    assert mapped.model == _BEDROCK_INVOKE  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # the public error's model


def _ssl_configuration(_ssl_verify: object) -> str:
    return "<ssl>"


@pytest.mark.parametrize(
    ("aiohttp", "provider", "origin", "rust_error", "text"),
    [
        (
            True,
            "bedrock",
            "http://127.0.0.1:9",
            f"could not reach the provider: {_TRANSPORT} (Connect): tcp connect error: Connection refused (os error 111)",
            "Cannot connect to host 127.0.0.1:9 ssl:<ssl> [Connect call failed ('127.0.0.1', 9)]",
        ),
        (
            True,
            "bedrock_mantle",
            "https://bedrock-mantle.us-east-1.api.aws:443",
            f"could not reach the provider: {_TRANSPORT} (Connect): dns error: error resolving DNS: "
            "failed to lookup address information: Name or service not known",
            "Cannot connect to host bedrock-mantle.us-east-1.api.aws:443 ssl:<ssl> [Name or service not known]",
        ),
        (
            False,
            "bedrock",
            "http://127.0.0.1:9",
            f"could not reach the provider: {_TRANSPORT} (Connect): tcp connect error: Connection refused (os error 111)",
            "All connection attempts failed",
        ),
    ],
    ids=("aiohttp-refused", "aiohttp-mantle-dns", "httpx"),
)
def test_a_bedrock_connect_failure_names_the_origin_rust_could_not_reach(
    monkeypatch: pytest.MonkeyPatch, aiohttp: bool, provider: str, origin: str, rust_error: str, text: str
) -> None:
    from litellm.rust_bridge.host import http_transport

    monkeypatch.setattr(http_transport, "uses_aiohttp_transport", lambda: aiohttp)
    monkeypatch.setattr(messages_host, "get_ssl_configuration", _ssl_configuration)
    unreachable: Final = RuntimeError(rust_error)
    unreachable.messages_connect_origin = origin  # pyright: ignore[reportAttributeAccessIssue]  # set by the Rust host
    model: Final = _BEDROCK_INVOKE if provider == "bedrock" else _BEDROCK_MANTLE

    mapped: Final = messages_host.map_failure(unreachable, {"model": model, "messages": [], **_AWS}, provider)

    assert type(mapped) is litellm.ServiceUnavailableError
    assert mapped.status_code == 503
    assert mapped.llm_provider == "bedrock"  # pyright: ignore[reportUnknownMemberType]  # the public error leaves llm_provider untyped
    assert text in str(mapped)


@pytest.mark.parametrize("aiohttp", (True, False), ids=("aiohttp", "httpx"))
def test_a_connect_failure_with_no_known_base_keeps_rust_s_text(monkeypatch: pytest.MonkeyPatch, aiohttp: bool) -> None:
    from litellm.rust_bridge.host import http_transport

    monkeypatch.setattr(http_transport, "uses_aiohttp_transport", lambda: aiohttp)
    rust_error: Final = (
        f"could not reach the provider: {_TRANSPORT} (Connect): tcp connect error: Connection refused (os error 111)"
    )

    mapped: Final = messages_host.map_failure(RuntimeError(rust_error), {"model": "claude-sonnet-4-5"}, "vertex_ai")

    assert type(mapped) is litellm.InternalServerError
    assert mapped.status_code == 500
    assert mapped.llm_provider == "vertex_ai"  # pyright: ignore[reportUnknownMemberType]  # the public error leaves llm_provider untyped
    assert mapped.message.endswith(f"Vertex_aiException InternalServerError - {rust_error}")


@pytest.mark.parametrize(
    ("provider", "api_base", "origin"),
    [
        (
            "bedrock_mantle",
            "http://bedrock-mantle.us-west-2.api.aws/v1",
            "https://bedrock-mantle.us-east-1.api.aws:443",
        ),
        ("bedrock_mantle", "", "https://bedrock-mantle.us-east-1.api.aws:443"),
        ("bedrock", "", "https://bedrock-runtime.us-east-1.amazonaws.com:443"),
    ],
    ids=("mantle-host-rebuilt-in-region", "mantle-blank-base", "invoke-blank-base"),
)
def test_a_bedrock_connect_failure_names_rusts_origin_over_the_callers_base(
    monkeypatch: pytest.MonkeyPatch, provider: str, api_base: str, origin: str
) -> None:
    from litellm.rust_bridge.host import http_transport

    monkeypatch.setattr(http_transport, "uses_aiohttp_transport", lambda: True)
    monkeypatch.setattr(messages_host, "get_ssl_configuration", _ssl_configuration)
    unreachable: Final = RuntimeError(
        f"could not reach the provider: {_TRANSPORT} (Connect): dns error: error resolving DNS: "
        "failed to lookup address information: Name or service not known"
    )
    unreachable.messages_connect_origin = origin  # pyright: ignore[reportAttributeAccessIssue]  # set by the Rust host
    model: Final = _BEDROCK_INVOKE if provider == "bedrock" else _BEDROCK_MANTLE

    mapped: Final = messages_host.map_failure(
        unreachable, {"model": model, "messages": [], "api_base": api_base, **_AWS}, provider
    )

    host: Final = origin.removeprefix("https://")
    assert type(mapped) is litellm.ServiceUnavailableError
    assert f"Cannot connect to host {host} ssl:<ssl> [Name or service not known]" in str(mapped)


@pytest.fixture
def no_aws_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_PROFILE",
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_ROLE_NAME",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", "/nonexistent")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", "/nonexistent")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")


def _credential_failure(kind: str) -> RuntimeError:
    native: Final = RuntimeError("provider authentication failed: could not load AWS default chain credentials")
    native.messages_credential_failure = kind  # pyright: ignore[reportAttributeAccessIssue]  # set by the native host
    return native


@pytest.mark.usefixtures("no_aws_credentials")
@pytest.mark.parametrize("kind", ("no_credentials", "replay"))
def test_a_bedrock_credential_failure_answers_401_with_what_python_signing_raises(kind: str) -> None:
    from botocore.exceptions import NoCredentialsError

    mapped: Final = messages_host.map_failure(
        _credential_failure(kind), {"model": _BEDROCK_INVOKE, "messages": [], "aws_region_name": "us-east-1"}, "bedrock"
    )

    assert type(mapped) is litellm.AuthenticationError
    assert mapped.status_code == 401
    assert type(mapped.__context__) is NoCredentialsError
    assert mapped.__traceback__ is not None


@pytest.mark.usefixtures("no_aws_credentials")
def test_a_bedrock_sso_profile_without_a_usable_token_answers_401(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config: Final = tmp_path / "config"
    config.write_text(
        "[profile logged-out-sso]\n"
        "sso_session = litellm-unit-test-session-never-logged-in\n"
        "sso_account_id = 111111111111\n"
        "sso_role_name = Dev\n"
        "[sso-session litellm-unit-test-session-never-logged-in]\n"
        "sso_start_url = https://example.awsapps.com/start\n"
        "sso_region = us-west-2\n"
    )
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config))

    mapped: Final = messages_host.map_failure(
        _credential_failure("replay"),
        {
            "model": _BEDROCK_INVOKE,
            "messages": [],
            "aws_region_name": "us-east-1",
            "aws_profile_name": "logged-out-sso",
        },
        "bedrock",
    )

    from botocore.exceptions import SSOTokenLoadError

    assert type(mapped) is litellm.AuthenticationError
    assert mapped.status_code == 401
    assert type(mapped.__context__) is SSOTokenLoadError


@pytest.mark.usefixtures("no_aws_credentials")
@pytest.mark.parametrize(
    ("sts_failure", "raised", "status"),
    (
        pytest.param(
            ("rejected", "AssumeRole", "AccessDenied", "denied"), "ClientError", None, id="assume-role-denied"
        ),
        pytest.param(
            ("rejected", "AssumeRoleWithWebIdentity", "ExpiredTokenException", "expired"),
            "ClientError",
            None,
            id="expired-token",
        ),
        pytest.param(("invalid_web_identity_token", "", "", "bad"), "AwsAuthError", 401, id="invalid-web-identity"),
        pytest.param(("unreachable", "", "", ""), "EndpointConnectionError", None, id="sts-unreachable"),
        pytest.param(None, "AuthenticationError", 401, id="no-sts-call"),
    ),
)
def test_a_bedrock_mantle_sts_failure_raises_the_class_the_mantle_signer_lets_through(
    sts_failure: tuple[str, str, str, str] | None, raised: str, status: int | None
) -> None:
    from botocore.exceptions import ClientError

    rust_error: Final = _credential_failure("replay")
    rust_error.messages_sts_failure = sts_failure  # pyright: ignore[reportAttributeAccessIssue]  # set by the Rust host

    mapped: Final = messages_host.map_failure(
        rust_error, {"model": _BEDROCK_MANTLE, "messages": [], "aws_region_name": "us-east-1"}, "bedrock_mantle"
    )

    assert type(mapped).__name__ == raised
    assert getattr(mapped, "status_code", None) == status
    assert mapped.__traceback__ is not None
    if isinstance(mapped, ClientError):
        assert sts_failure is not None
        assert mapped.response.get("Error", {}).get("Code") == sts_failure[2]
        assert mapped.operation_name == sts_failure[1]


@pytest.mark.usefixtures("no_aws_credentials")
def test_an_access_key_without_its_secret_answers_401_with_the_partial_credentials_error_python_signing_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from botocore.exceptions import PartialCredentialsError

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")

    mapped: Final = messages_host.map_failure(
        _credential_failure("no_credentials"),
        {"model": _BEDROCK_INVOKE, "messages": [], "aws_region_name": "us-east-1"},
        "bedrock",
    )

    assert type(mapped) is litellm.AuthenticationError
    assert type(mapped.__context__) is PartialCredentialsError
    assert getattr(mapped.__context__, "kwargs", None) == {"provider": "env", "cred_var": "AWS_SECRET_ACCESS_KEY"}


@pytest.mark.usefixtures("no_aws_credentials")
@pytest.mark.parametrize(
    ("token", "kind", "status", "text"),
    [
        (
            "os.environ/AWS_WEB_IDENTITY_TOKEN_REFERENCE",
            "web_identity_reference",
            400,
            "Invalid web identity token reference.",
        ),
        (
            "oidc-secret-that-is-not-set",
            "missing_web_identity_token",
            401,
            "OIDC token could not be retrieved from secret manager.",
        ),
    ],
    ids=("environment-reference", "unresolved-token"),
)
@pytest.mark.parametrize("replayed", (False, True), ids=("typed", "replayed"))
def test_a_bedrock_web_identity_failure_raises_the_aws_auth_error_python_raises(
    token: str, kind: str, status: int, text: str, replayed: bool
) -> None:
    from litellm.llms.bedrock.base_aws_llm import AwsAuthError

    request: Final[dict[str, object]] = {
        "model": _BEDROCK_INVOKE,
        "messages": [],
        "aws_region_name": "us-east-1",
        "aws_web_identity_token": token,
        "aws_role_name": "arn:aws:iam::123456789012:role/bedrock",
        "aws_session_name": "litellm-session",
    }

    mapped: Final = messages_host.map_failure(_credential_failure("replay" if replayed else kind), request, "bedrock")

    assert type(mapped) is AwsAuthError
    assert mapped.status_code == status
    assert str(mapped) == text


def test_a_bedrock_credential_failure_python_would_sign_through_keeps_the_native_mapping() -> None:
    mapped: Final = messages_host.map_failure(
        _credential_failure("replay"), {"model": _BEDROCK_INVOKE, "messages": [], **_AWS}, "bedrock"
    )

    assert isinstance(mapped, litellm.APIConnectionError | litellm.InternalServerError)
    assert mapped.llm_provider == "bedrock"  # pyright: ignore[reportUnknownMemberType]  # the public error leaves llm_provider untyped


@pytest.mark.usefixtures("no_aws_credentials")
def test_a_credential_failure_text_without_the_marker_keeps_the_native_mapping() -> None:
    native: Final = RuntimeError("credential acquisition failed: no credentials in the default chain")

    mapped: Final = messages_host.map_failure(
        native, {"model": _BEDROCK_INVOKE, "messages": [], "aws_region_name": "us-east-1"}, "bedrock"
    )

    assert isinstance(mapped, litellm.APIConnectionError | litellm.InternalServerError)
    assert mapped.llm_provider == "bedrock"  # pyright: ignore[reportUnknownMemberType]  # the public error leaves llm_provider untyped


_VERTEX_CLAUDE: Final = "vertex_ai/claude-sonnet-4-5"


def test_a_vertex_connect_failure_without_an_api_base_keeps_rust_s_text(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.rust_bridge.host import http_transport

    monkeypatch.setattr(http_transport, "uses_aiohttp_transport", lambda: True)
    native: Final = (
        f"could not reach the provider: {_TRANSPORT} (Connect): dns error: "
        "error resolving DNS: failed to lookup address information: Name or service not known"
    )

    mapped: Final = messages_host.map_failure(
        RuntimeError(native),
        {"model": _VERTEX_CLAUDE, "messages": [], "custom_llm_provider": "vertex_ai", "vertex_location": "elsewhere"},
        "vertex_ai",
    )

    assert type(mapped) is litellm.InternalServerError
    assert mapped.llm_provider == "vertex_ai"  # pyright: ignore[reportUnknownMemberType]  # the public error leaves llm_provider untyped
    assert str(mapped).endswith(native)


_TLS_CLOSED: Final = (
    f"upstream network error: {_TRANSPORT} (SendRequest): connection error: peer closed connection without "
    "sending TLS close_notify: https://docs.rs/rustls/latest/rustls/manual/_03_howto/index.html#unexpected-eof"
)


@pytest.mark.parametrize(
    ("aiohttp", "provider", "model", "text"),
    [
        (True, "vertex_ai", _VERTEX_CLAUDE, "Server disconnected"),
        (False, "vertex_ai", _VERTEX_CLAUDE, "Server disconnected without sending a response."),
        (True, "anthropic", _ADAPTIVE_OPUS, "peer closed connection without sending TLS close_notify"),
    ],
    ids=("vertex_aiohttp", "vertex_httpx", "anthropic_keeps_main_s_text"),
)
def test_a_tls_close_before_headers_reads_as_a_disconnect_on_vertex_ai_only(
    monkeypatch: pytest.MonkeyPatch, aiohttp: bool, provider: str, model: str, text: str
) -> None:
    from litellm.litellm_core_utils.error_normalization import normalize_error
    from litellm.rust_bridge.host import http_transport

    monkeypatch.setattr(http_transport, "uses_aiohttp_transport", lambda: aiohttp)

    mapped: Final = messages_host.map_failure(
        RuntimeError(_TLS_CLOSED),
        {"model": model, "messages": [], "custom_llm_provider": provider, "vertex_project": "p"},
        provider,
    )

    assert type(mapped) is litellm.InternalServerError
    assert text in str(mapped)
    expected: Final = "500_PROVIDER_CONNECTION_ERROR" if provider == "anthropic" else "500_PROVIDER_INTERNAL_ERROR"
    assert normalize_error(mapped, "500", str(mapped)) == expected


def _google_login_failure(name: str, message: str) -> RuntimeError:
    raised: Final = RuntimeError("provider authentication failed: " + message)
    raised.messages_google_login_failure = (name, message)  # pyright: ignore[reportAttributeAccessIssue]  # set by the Rust host
    return raised


def _vertex_logger() -> Logging:
    return Logging(
        model=_VERTEX_CLAUDE,
        messages=[],
        stream=False,
        call_type="aanthropic_messages",
        start_time=None,
        litellm_call_id="call-1",
        function_id="fn",
    )


@pytest.mark.parametrize(
    ("name", "error_class"),
    [
        ("ValueError", ValueError),
        ("OSError", OSError),
        ("google.auth.exceptions.DefaultCredentialsError", google_auth.DefaultCredentialsError),
        ("google.auth.exceptions.InvalidResource", google_auth.InvalidResource),
        ("google.auth.exceptions.MalformedError", google_auth.MalformedError),
        ("google.auth.exceptions.InvalidValue", google_auth.InvalidValue),
        ("google.auth.exceptions.RefreshError", google_auth.RefreshError),
        ("google.auth.exceptions.TransportError", google_auth.TransportError),
        ("google.auth.exceptions.OAuthError", google_auth.OAuthError),
    ],
)
def test_a_vertex_login_failure_raises_the_exception_python_s_load_auth_raises(
    name: str, error_class: type[Exception]
) -> None:
    logger: Final = _vertex_logger()

    mapped: Final = messages_host.map_failure(
        _google_login_failure(name, "Could not resolve project_id"),
        {"model": _VERTEX_CLAUDE, "messages": [], "litellm_logging_obj": logger},
        "vertex_ai",
        True,
    )

    assert type(mapped) is error_class
    assert str(mapped) == "Could not resolve project_id"
    assert mapped.__traceback__ is not None
    assert logger.model_call_details["litellm_params"]["aanthropic_messages"] is True


def test_a_vertex_credentials_file_failure_is_raised_as_the_file_failure_rust_names() -> None:
    message: Final = "Unable to load the vertex credentials file. LiteLLM's log names the path and the reason."

    mapped: Final = messages_host.map_failure(
        _google_login_failure("ValueError", message),
        {"model": _VERTEX_CLAUDE, "messages": [], "litellm_logging_obj": _vertex_logger()},
        "vertex_ai",
    )

    assert type(mapped) is ValueError
    assert str(mapped) == message


def test_a_google_login_failure_outside_vertex_ai_keeps_the_rust_mapping() -> None:
    mapped: Final = messages_host.map_failure(
        _google_login_failure("ValueError", "Could not resolve project_id"),
        {"model": _BEDROCK_INVOKE, "messages": [], **_AWS},
        "bedrock",
    )

    assert not isinstance(mapped, ValueError)


@pytest.mark.parametrize(
    "name", ["TypeError", "google.auth.exceptions.GoogleAuthError", "google.auth.exceptions.__class__", "SystemExit"]
)
def test_a_vertex_login_failure_class_outside_the_allowlist_keeps_the_rust_mapping(name: str) -> None:
    mapped: Final = messages_host.map_failure(
        _google_login_failure(name, "Could not resolve project_id"),
        {"model": _VERTEX_CLAUDE, "messages": [], "litellm_logging_obj": _vertex_logger()},
        "vertex_ai",
    )

    assert not isinstance(mapped, TypeError | google_auth.GoogleAuthError | SystemExit)
    assert str(mapped) != "Could not resolve project_id"


@pytest.mark.parametrize(
    ("thread", "error_class", "status", "text"),
    [
        (
            {"type": "create"},
            litellm.BadRequestError,
            400,
            'litellm.BadRequestError: BedrockException - {"type": "error", "error": {"type": "invalid_request_error", '
            '"message": "message threads are not supported for provider bedrock. Send the request without thread, '
            'or set drop_params to serve it without a server-side thread", '
            '"details": {"error_code": "thread_unsupported_request"}}}',
        ),
        (
            {"type": "continue", "previous_message_id": "msg_1"},
            litellm.NotFoundError,
            404,
            'litellm.NotFoundError: BedrockException - {"type": "error", "error": {"type": "not_found_error", '
            '"message": "message threads are not supported for provider bedrock, so this thread cannot be continued. '
            'Send the full conversation to start over", "details": {"error_code": "thread_not_found"}}}',
        ),
    ],
    ids=("create", "continue"),
)
def test_settings_refuse_a_thread_bedrock_cannot_serve_as_the_python_handler_does(
    monkeypatch: pytest.MonkeyPatch,
    thread: Mapping[str, str],
    error_class: type[Exception],
    status: int,
    text: str,
) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    request: Final = {**_CALL, "model": _BEDROCK_INVOKE, "thread": dict(thread), **_AWS}

    with pytest.raises(error_class) as raised:
        messages_host.settings({}, request, asynchronous=True)

    assert str(raised.value) == text
    assert raised.value.status_code == status  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # the public error's status
    assert raised.value.llm_provider == "bedrock"  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # the public error's provider


def test_a_thread_bedrock_cannot_serve_is_dropped_under_drop_params() -> None:
    request: Final = {**_CALL, "model": _BEDROCK_INVOKE, "thread": {"type": "create"}, "drop_params": True, **_AWS}

    messages_host.settings({}, request, asynchronous=True)

    assert "thread" not in messages_host.body_params(request, asynchronous=True)


def test_a_bedrock_thread_continue_is_not_found_even_under_drop_params() -> None:
    request: Final = {
        **_CALL,
        "model": _BEDROCK_INVOKE,
        "thread": {"type": "continue", "previous_message_id": "msg_1"},
        "drop_params": True,
        **_AWS,
    }

    with pytest.raises(litellm.NotFoundError):
        messages_host.settings({}, request, asynchronous=True)


@pytest.mark.parametrize(
    ("api_base", "native", "sent"),
    [
        (
            " https://bedrock.example ",
            "https://bedrock.example/model/m/invoke",
            " https://bedrock.example /model/m/invoke",
        ),
        ("https://bedrock.example", "https://bedrock.example/model/m/invoke", "https://bedrock.example/model/m/invoke"),
    ],
    ids=("untrimmed", "clean"),
)
def test_a_bedrock_call_sends_the_url_python_joins_onto_the_callers_api_base(
    api_base: str, native: str, sent: str
) -> None:
    assert messages_host._sent_url({"model": _BEDROCK_INVOKE, "api_base": api_base}, native) == sent  # pyright: ignore[reportPrivateUsage]  # the route's own URL fix-up


def test_record_guardrail_bills_the_bedrock_trace_on_the_calls_logger() -> None:
    logger: Final = Logging(
        model=_BEDROCK_INVOKE,
        messages=[],
        stream=False,
        call_type="anthropic_messages",
        start_time=datetime.datetime.now(),
        litellm_call_id="guardrail",
        function_id="guardrail",
    )
    logger.update_from_kwargs(  # pyright: ignore[reportUnknownMemberType]  # the logger's untyped dict params
        kwargs={},
        model=_BEDROCK_INVOKE,
        optional_params={"guardrailConfig": {"guardrailIdentifier": "gr-1", "guardrailVersion": "1"}},
        litellm_params={"aws_region_name": "us-east-1"},
        custom_llm_provider="bedrock",
    )
    trace: Final[dict[str, object]] = {
        "guardrail": {
            "input": {"gr-1": {"invocationMetrics": {"usage": {"contentPolicyUnits": 1}}}},
            "outputs": [],
        }
    }

    messages_host.record_guardrail(
        {"id": "msg_1", "type": "message", "content": [], "amazon-bedrock-trace": trace},
        {"model": _BEDROCK_INVOKE, "litellm_logging_obj": logger},
    )

    metadata: Final = logger.model_call_details["litellm_params"]["metadata"]  # pyright: ignore[reportAny]  # the logger's untyped details
    assert metadata["standard_logging_guardrail_information"]


def test_a_bedrock_web_search_rejection_raises_the_bad_request_the_invoke_config_raises() -> None:
    text: Final = "Bedrock does not support Anthropic's server-side web search tool"
    native: Final = ValueError(text)
    native.messages_request_error = True  # pyright: ignore[reportAttributeAccessIssue]  # set by the native host
    native.messages_provider_rejection = text  # pyright: ignore[reportAttributeAccessIssue]  # set by the native host
    native.messages_provider_bad_request = True  # pyright: ignore[reportAttributeAccessIssue]  # set by the native host

    mapped: Final = messages_host.map_failure(native, {"model": _BEDROCK_INVOKE, "messages": [], **_AWS}, "bedrock")

    assert type(mapped) is litellm.BadRequestError
    assert str(mapped) == f"litellm.BadRequestError: {text}"
    assert mapped.llm_provider == "bedrock"  # pyright: ignore[reportUnknownMemberType]  # the public error leaves llm_provider untyped
    assert mapped.model == "us.anthropic.claude-sonnet-4-5-20250929-v1:0"  # pyright: ignore[reportUnknownMemberType]  # the public error leaves model untyped


_BEDROCK_MANTLE: Final = "bedrock_mantle/anthropic.claude-sonnet-5-5"
_MANTLE_ROUTE: Final = "bedrock/mantle/anthropic.claude-sonnet-5-5"


_WEB_SEARCH_REFUSAL: Final = "Bedrock does not support Anthropic's server-side web search tool"


@pytest.mark.parametrize(
    ("named_provider", "text"),
    (
        ("bedrock", f"litellm.BadRequestError: {_WEB_SEARCH_REFUSAL}"),
        # Unnamed, the host maps it as any config's HTTP handler maps the transform's error.
        (None, f"litellm.BadRequestError: BedrockException - litellm.BadRequestError: {_WEB_SEARCH_REFUSAL}"),
    ),
    ids=("rust-names-bedrock", "rust-names-none"),
)
def test_a_mantle_bad_request_rejection_raises_as_the_provider_rust_names(
    named_provider: str | None, text: str
) -> None:
    native: Final = ValueError(_WEB_SEARCH_REFUSAL)
    native.messages_request_error = True  # pyright: ignore[reportAttributeAccessIssue]  # set by the Rust host
    native.messages_provider_rejection = _WEB_SEARCH_REFUSAL  # pyright: ignore[reportAttributeAccessIssue]  # set by the Rust host
    native.messages_provider_bad_request = True  # pyright: ignore[reportAttributeAccessIssue]  # set by the Rust host
    native.messages_provider_model = "anthropic.claude-sonnet-5-5"  # pyright: ignore[reportAttributeAccessIssue]  # set by the Rust host
    if named_provider is not None:
        native.messages_provider = named_provider  # pyright: ignore[reportAttributeAccessIssue]  # set by the Rust host

    mapped: Final = messages_host.map_failure(native, {"model": _BEDROCK_MANTLE, "messages": []}, "bedrock_mantle")

    assert type(mapped) is litellm.BadRequestError
    assert mapped.status_code == 400
    assert str(mapped) == text
    assert mapped.llm_provider == "bedrock"  # pyright: ignore[reportUnknownMemberType]  # the public error leaves llm_provider untyped


@pytest.mark.parametrize(
    ("model", "provider", "named_model", "raised_model"),
    (
        (_BEDROCK_MANTLE, "bedrock_mantle", "anthropic.claude-sonnet-5-5", "anthropic.claude-sonnet-5-5"),
        (_MANTLE_ROUTE, "bedrock", "anthropic.claude-sonnet-5-5", "anthropic.claude-sonnet-5-5"),
        (_BEDROCK_MANTLE, "bedrock_mantle", None, "anthropic.claude-sonnet-5-5"),
    ),
    ids=("bedrock-mantle", "mantle-route", "no-named-model"),
)
def test_a_refused_reply_raises_an_api_error_500(
    model: str, provider: str, named_model: str | None, raised_model: str
) -> None:
    text: Final = "invalid response: Bedrock Mantle returned a `usage` litellm cannot bill: got an array"
    refused: Final = RuntimeError(text)
    refused.messages_refused_reply = True  # pyright: ignore[reportAttributeAccessIssue]  # set by the Rust host
    if named_model is not None:
        refused.messages_provider_model = named_model  # pyright: ignore[reportAttributeAccessIssue]  # set by the Rust host

    mapped: Final = messages_host.map_failure(refused, {"model": model, "messages": []}, provider)

    assert type(mapped) is litellm.APIError
    assert mapped.status_code == 500
    assert mapped.llm_provider == provider  # pyright: ignore[reportUnknownMemberType]  # the public error leaves llm_provider untyped
    assert mapped.model == raised_model  # pyright: ignore[reportUnknownMemberType]  # the public error leaves model untyped
    assert text in str(mapped)


@pytest.mark.usefixtures("no_aws_credentials")
def test_a_bedrock_mantle_credential_failure_answers_401_with_the_mantle_auth_error() -> None:
    from litellm.llms.bedrock_mantle.common_utils import BedrockMantleAuthError

    mapped: Final = messages_host.map_failure(
        _credential_failure("no_credentials"),
        {"model": _BEDROCK_MANTLE, "messages": [], "api_base": "https://mantle.test"},
        "bedrock_mantle",
    )

    assert type(mapped) is litellm.AuthenticationError
    assert mapped.status_code == 401
    assert type(mapped.__context__) is BedrockMantleAuthError
    assert mapped.__traceback__ is not None


@pytest.mark.usefixtures("no_aws_credentials")
@pytest.mark.parametrize(("kind", "status"), (("web_identity_reference", 400), ("missing_web_identity_token", 401)))
def test_a_bedrock_mantle_web_identity_failure_keeps_the_aws_auth_error(kind: str, status: int) -> None:
    from litellm.llms.bedrock.base_aws_llm import AwsAuthError

    mapped: Final = messages_host.map_failure(
        _credential_failure(kind), {"model": _BEDROCK_MANTLE, "messages": []}, "bedrock_mantle"
    )

    assert type(mapped) is AwsAuthError
    assert mapped.status_code == status


@pytest.mark.usefixtures("no_aws_credentials")
def test_a_bedrock_mantle_route_credential_failure_answers_401() -> None:
    from botocore.exceptions import NoCredentialsError

    mapped: Final = messages_host.map_failure(
        _credential_failure("no_credentials"),
        {"model": _MANTLE_ROUTE, "messages": [], "api_base": "https://mantle.test"},
        "bedrock",
    )

    assert type(mapped) is litellm.AuthenticationError
    assert type(mapped.__context__) is NoCredentialsError


@pytest.mark.parametrize(
    ("model", "request_provider", "raised_by_signing"),
    (
        pytest.param(_MANTLE_ROUTE, "bedrock", "ProfileNotFound", id="mantle-route"),
        pytest.param(_BEDROCK_MANTLE, "bedrock_mantle", "BedrockMantleAuthError", id="bedrock-mantle"),
    ),
)
def test_a_mantle_missing_profile_answers_401_with_what_python_signing_raises(
    model: str, request_provider: str, raised_by_signing: str
) -> None:
    rust_error: Final = _credential_failure("profile_not_found")
    rust_error.messages_missing_profile = "absent-profile"  # pyright: ignore[reportAttributeAccessIssue]  # set by the Rust host

    mapped: Final = messages_host.map_failure(
        rust_error, {"model": model, "messages": [], "aws_region_name": "us-east-1"}, request_provider
    )

    assert type(mapped) is litellm.AuthenticationError
    assert type(mapped.__context__).__name__ == raised_by_signing


def _refused(text: str, **markers: object) -> ValueError:
    rust_error: Final = ValueError(text)
    rust_error.messages_request_error = True  # pyright: ignore[reportAttributeAccessIssue]  # set by the Rust host
    for name, value in markers.items():
        setattr(rust_error, name, value)
    return rust_error


def test_a_region_refused_while_resolving_bedrock_mantle_raises_a_bad_request() -> None:
    mapped: Final = messages_host.map_failure(
        _refused("bad region"), {"model": _BEDROCK_MANTLE, "messages": []}, "bedrock_mantle"
    )

    assert type(mapped) is litellm.BadRequestError
    assert mapped.status_code == 400


def test_a_bedrock_mantle_web_search_rejection_raises_the_invoke_transforms_bad_request() -> None:
    text: Final = "Bedrock does not support Anthropic's server-side web search tool"
    rust_error: Final = _refused(text)
    rust_error.messages_provider_rejection = text  # pyright: ignore[reportAttributeAccessIssue]  # set by the Rust host
    rust_error.messages_provider_bad_request = True  # pyright: ignore[reportAttributeAccessIssue]  # set by the Rust host
    rust_error.messages_provider_model = "anthropic.claude-sonnet-5-5"  # pyright: ignore[reportAttributeAccessIssue]  # set by the Rust host
    rust_error.messages_provider = "bedrock"  # pyright: ignore[reportAttributeAccessIssue]  # set by the Rust host

    mapped: Final = messages_host.map_failure(
        rust_error, {"model": "bedrock_mantle/us-west-2/anthropic.claude-sonnet-5-5", "messages": []}, "bedrock_mantle"
    )

    assert type(mapped) is litellm.BadRequestError
    assert str(mapped) == f"litellm.BadRequestError: {text}"
    assert mapped.llm_provider == "bedrock"  # pyright: ignore[reportUnknownMemberType]  # the public error leaves llm_provider untyped
    assert mapped.model == "anthropic.claude-sonnet-5-5"  # pyright: ignore[reportUnknownMemberType]  # the public error leaves model untyped


@pytest.mark.parametrize(
    ("model", "shapes_invoke_tools", "shaped"),
    (
        pytest.param(_BEDROCK_MANTLE, False, False, id="bedrock-mantle"),
        pytest.param(_MANTLE_ROUTE, False, False, id="mantle-route"),
        pytest.param(_BEDROCK_INVOKE, True, True, id="bedrock-invoke"),
    ),
)
def test_body_params_shape_caller_tools_only_when_rust_asks_and_drop_a_body_anthropic_beta(
    model: str, shapes_invoke_tools: bool, shaped: bool
) -> None:
    tools: Final = [{"type": "custom", "name": "", "description": "d", "input_schema": {"type": "custom"}}]
    request: Final[Mapping[str, object]] = {
        "model": model,
        "messages": [],
        "max_tokens": 16,
        "tools": tools,
        "anthropic_beta": ["context-management-2025-06-27"],
    }

    params: Final = messages_host.body_params(request, asynchronous=True, shapes_invoke_tools=shapes_invoke_tools)

    assert "anthropic_beta" not in params
    assert (
        tools
        == [
            {"type": "custom", "name": "litellm_unnamed_tool_0", "description": "d", "input_schema": {"type": "object"}}
        ]
    ) is shaped


@pytest.mark.parametrize(
    "request_fields",
    (
        pytest.param(
            {"model": _BEDROCK_MANTLE, "api_key": "bearer-key", "aws_region_name": "us-west-2"},
            id="python-builds-and-signs-it",
        ),
        pytest.param({"model": _BEDROCK_MANTLE, "aws_region_name": "US_WEST_2"}, id="no-credentials-bad-region"),
    ),
)
@pytest.mark.usefixtures("no_aws_credentials")
def test_a_mantle_refusal_python_would_send_keeps_the_native_bad_request(
    request_fields: Mapping[str, object],
) -> None:
    mapped: Final = messages_host.map_failure(
        _refused("rust refusal"), {**request_fields, "messages": []}, "bedrock_mantle"
    )

    assert type(mapped) is litellm.BadRequestError
    assert mapped.status_code == 400
    assert mapped.llm_provider == "bedrock_mantle"  # pyright: ignore[reportUnknownMemberType]  # the public error leaves llm_provider untyped


def _mantle_logger() -> Logging:
    return Logging(
        model=_MANTLE_ROUTE,
        messages=[],
        stream=False,
        call_type="anthropic_messages",
        start_time=datetime.datetime.now(),
        litellm_call_id="mantle-failure",
        function_id="mantle-failure",
    )


def test_a_bedrock_mantle_region_refused_resolving_the_provider_names_no_provider() -> None:
    rust_error: Final = _refused(
        "Invalid AWS region format: 'US-EAST-1'.", messages_provider="", messages_provider_model=_BEDROCK_MANTLE
    )

    mapped: Final = messages_host.map_failure(rust_error, {"model": _BEDROCK_MANTLE, "messages": []}, "bedrock_mantle")

    assert type(mapped) is litellm.BadRequestError
    assert mapped.status_code == 400
    assert mapped.llm_provider == ""  # pyright: ignore[reportUnknownMemberType]  # the public error leaves llm_provider untyped
    assert mapped.model == _BEDROCK_MANTLE  # pyright: ignore[reportUnknownMemberType]  # the public error leaves model untyped


def test_a_mantle_route_refusal_logs_the_route_model_and_raises_the_model_id_rust_names() -> None:
    """Python's handler logs the route model before the Mantle config's transform refuses the call
    with the model id it hands the Bedrock invoke transform."""
    text: Final = "Bedrock does not support Anthropic's server-side web search tool"
    rust_error: Final = _refused(
        text,
        messages_provider_rejection=text,
        messages_provider_bad_request=True,
        messages_provider_model="anthropic.claude-sonnet-5-5",
    )
    logger: Final = _mantle_logger()

    mapped: Final = messages_host.map_failure(
        rust_error, {"model": _MANTLE_ROUTE, "messages": [], "litellm_logging_obj": logger}, "bedrock"
    )

    assert type(mapped) is litellm.BadRequestError
    assert mapped.model == "anthropic.claude-sonnet-5-5"  # pyright: ignore[reportUnknownMemberType]  # the public error leaves model untyped
    assert logger.model == "mantle/anthropic.claude-sonnet-5-5"


@pytest.mark.usefixtures("no_aws_credentials")
def test_a_mantle_route_credential_failure_logs_the_model_id_rust_names() -> None:
    rust_error: Final = _credential_failure("no_credentials")
    rust_error.messages_provider_model = "anthropic.claude-sonnet-5-5"  # pyright: ignore[reportAttributeAccessIssue]  # set by the Rust host
    logger: Final = _mantle_logger()

    messages_host.map_failure(
        rust_error, {"model": _MANTLE_ROUTE, "messages": [], "litellm_logging_obj": logger}, "bedrock"
    )

    assert logger.model == "anthropic.claude-sonnet-5-5"


@pytest.mark.usefixtures("no_aws_credentials")
@pytest.mark.parametrize(
    ("sts_failure", "raised"),
    (
        pytest.param(None, "AuthenticationError", id="no-credentials"),
        pytest.param(("rejected", "AssumeRole", "AccessDenied", "denied"), "ClientError", id="assume-role-denied"),
        pytest.param(("unreachable", "", "", ""), "EndpointConnectionError", id="sts-unreachable"),
    ),
)
def test_a_mantle_route_credential_failure_maps_the_kind_rust_names(
    sts_failure: tuple[str, str, str, str] | None, raised: str
) -> None:
    rust_error: Final = _credential_failure("unavailable")
    rust_error.messages_sts_failure = sts_failure  # pyright: ignore[reportAttributeAccessIssue]  # set by the Rust host

    mapped: Final = messages_host.map_failure(
        rust_error, {"model": _MANTLE_ROUTE, "messages": [], "aws_region_name": "us-east-1"}, "bedrock"
    )

    assert type(mapped).__name__ == raised
    assert mapped.__traceback__ is not None


_DIAGNOSTICS: Final = {"previous_message_id": "msg_01previous"}
_JSON_OBJECT: Final = TypeAdapter(dict[str, object])
_AWS_TEST_CREDENTIALS: Final = {
    "aws_access_key_id": "test-access-key",
    "aws_secret_access_key": "test-secret-key",
    "aws_region_name": "us-east-1",
}


@pytest.fixture
def recording_upstream() -> Iterator[tuple[str, list[dict[str, object]]]]:
    bodies: Final[list[dict[str, object]]] = []
    answer: Final = json.dumps(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 3, "output_tokens": 1},
        }
    ).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            bodies.append(_JSON_OBJECT.validate_json(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(answer)))
            self.end_headers()
            self.wfile.write(answer)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server: Final = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", bodies
    server.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("drop_params", (False, True), ids=("default", "drop_params"))
async def test_anthropic_messages_forwards_diagnostics_to_anthropic(
    recording_upstream: tuple[str, list[dict[str, object]]], drop_params: bool
) -> None:
    api_base, bodies = recording_upstream

    await litellm.anthropic_messages(
        model="anthropic/claude-opus-5-5",
        api_base=api_base,
        api_key="sk-ant-test",
        max_tokens=16,
        messages=[{"role": "user", "content": "hi"}],
        diagnostics=_DIAGNOSTICS,
        drop_params=drop_params,
    )

    assert bodies[0]["diagnostics"] == _DIAGNOSTICS


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "deployment",
    (
        {"model": "bedrock/invoke/us.anthropic.claude-opus-4-8", **_AWS_TEST_CREDENTIALS},
        {"model": "bedrock/mantle/anthropic.claude-mythos-preview", **_AWS_TEST_CREDENTIALS},
    ),
    ids=("bedrock_invoke", "bedrock_mantle"),
)
async def test_anthropic_messages_refuses_diagnostics_a_provider_rejects_without_drop_params(
    recording_upstream: tuple[str, list[dict[str, object]]],
    deployment: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    api_base, bodies = recording_upstream

    with pytest.raises(litellm.UnsupportedParamsError, match="drop_params") as refused:
        await litellm.anthropic_messages(
            api_base=api_base,
            max_tokens=16,
            messages=[{"role": "user", "content": "hi"}],
            diagnostics=_DIAGNOSTICS,
            **deployment,
        )

    assert refused.value.status_code == 400
    assert bodies == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "drop",
    ({"drop_params": True}, {"additional_drop_params": ["diagnostics"]}),
    ids=("drop_params", "additional_drop_params"),
)
async def test_anthropic_messages_drops_diagnostics_a_provider_rejects_when_asked(
    recording_upstream: tuple[str, list[dict[str, object]]],
    drop: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    api_base, bodies = recording_upstream

    response: Final = await litellm.anthropic_messages(
        model="bedrock/invoke/us.anthropic.claude-opus-4-8",
        api_base=api_base,
        max_tokens=16,
        messages=[{"role": "user", "content": "hi"}],
        diagnostics=_DIAGNOSTICS,
        **_AWS_TEST_CREDENTIALS,
        **drop,
    )

    assert _JSON_OBJECT.validate_python(response)["id"] == "msg_test"
    assert "diagnostics" not in bodies[0]


@pytest.fixture
def unavailable_container_credentials(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(503)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            pass

    server: Final = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("AWS_CONTAINER_CREDENTIALS_FULL_URI", f"http://127.0.0.1:{server.server_port}/credentials")
    yield
    server.shutdown()


@pytest.mark.usefixtures("no_aws_credentials", "unavailable_container_credentials")
@pytest.mark.parametrize("kind", ("unavailable", "replay"))
def test_a_bedrock_mantle_credential_source_outage_stays_the_retryable_mantle_value_error(kind: str) -> None:
    from litellm.llms.bedrock_mantle.common_utils import MANTLE_AUTH_FAILED

    mapped: Final = messages_host.map_failure(
        _credential_failure(kind),
        {"model": _BEDROCK_MANTLE, "messages": [], "aws_region_name": "us-east-1"},
        "bedrock_mantle",
    )

    assert type(mapped) is ValueError
    assert str(mapped) == MANTLE_AUTH_FAILED


@pytest.mark.usefixtures("no_aws_credentials")
def test_a_bedrock_mantle_sso_profile_without_a_usable_token_answers_401(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from botocore.exceptions import SSOTokenLoadError

    config: Final = tmp_path / "config"
    config.write_text(
        "[profile logged-out-sso]\n"
        "sso_session = litellm-unit-test-session-never-logged-in\n"
        "sso_account_id = 111111111111\n"
        "sso_role_name = Dev\n"
        "[sso-session litellm-unit-test-session-never-logged-in]\n"
        "sso_start_url = https://example.awsapps.com/start\n"
        "sso_region = us-west-2\n"
    )
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config))

    mapped: Final = messages_host.map_failure(
        _credential_failure("unavailable"),
        {
            "model": _BEDROCK_MANTLE,
            "messages": [],
            "aws_region_name": "us-east-1",
            "aws_profile_name": "logged-out-sso",
        },
        "bedrock_mantle",
    )

    assert type(mapped) is litellm.AuthenticationError
    assert type(mapped.__context__) is SSOTokenLoadError


@pytest.mark.usefixtures("no_aws_credentials", "unavailable_container_credentials")
def test_a_mantle_route_credential_source_outage_stays_the_retryable_credential_error() -> None:
    from botocore.exceptions import CredentialRetrievalError

    mapped: Final = messages_host.map_failure(
        _credential_failure("unavailable"),
        {"model": _MANTLE_ROUTE, "messages": [], "aws_region_name": "us-east-1"},
        "bedrock",
    )

    assert type(mapped) is CredentialRetrievalError


def _priced_logger(litellm_params: Mapping[str, object]) -> Logging:
    logger: Final = Logging(
        model="claude-opus-5-5",
        messages=[],
        stream=False,
        call_type="anthropic_messages",
        start_time=datetime.datetime.now(),
        litellm_call_id="priced",
        function_id="priced",
    )
    logger.litellm_params = dict(litellm_params)
    logger.model_call_details["litellm_params"] = logger.litellm_params
    return logger


_ROUTED: Final = MappingProxyType({"metadata": {"model_info": {"id": "deployment-1", "base_model": "claude-opus-5-5"}}})


@pytest.mark.parametrize(
    ("model", "litellm_params", "expected"),
    [
        (
            "anthropic/claude-opus-5-5",
            _ROUTED,
            {
                "deployment_id": "deployment-1",
                "base_model": "claude-opus-5-5",
                "region": None,
                "vertex_location": None,
                "data_residency": None,
            },
        ),
        (
            "vertex_ai/claude-opus-5-5",
            {"vertex_location": "us-east5", "data_residency": "EU"},
            {
                "deployment_id": None,
                "base_model": None,
                "region": None,
                "vertex_location": "us-east5",
                "data_residency": "EU",
            },
        ),
        (
            _MANTLE_ROUTE,
            {"aws_region_name": "us-gov-west-1"},
            {
                "deployment_id": None,
                "base_model": None,
                "region": "us-gov-west-1",
                "vertex_location": None,
                "data_residency": None,
            },
        ),
        ("anthropic/claude-opus-5-5", {**_ROUTED, "input_cost_per_token": 1e-6, "output_cost_per_token": 2e-6}, None),
        (
            "anthropic/claude-opus-5-5",
            {"metadata": {"model_info": {"id": "deployment-1", "pricing_provider": "bedrock"}}},
            None,
        ),
    ],
)
def test_settings_hand_the_route_the_deployment_facts_its_logging_would_price_with(
    model: str, litellm_params: Mapping[str, object], expected: Mapping[str, object] | None
) -> None:
    logger: Final = _priced_logger(litellm_params)

    settings: Final = messages_host.settings({"litellm_logging_obj": logger}, {**_CALL, "model": model})

    assert settings["pricing"] == expected


def test_the_route_prices_at_the_routed_deployments_vertex_location_before_the_logger_holds_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "vertex_location", "global")
    logger: Final = _priced_logger({})
    request: Final = {**_CALL, "vertex_location": "us-east5", "model": "vertex_ai/claude-opus-5-5"}

    priced_at: Final = messages_host.settings({"litellm_logging_obj": logger}, request)["pricing"]

    assert isinstance(priced_at, dict)
    assert priced_at["vertex_location"] == "us-east5"


@pytest.mark.parametrize("deployment_location", ["us-east5", None])
def test_the_route_prices_at_the_vertex_location_its_logging_would_price_at(
    monkeypatch: pytest.MonkeyPatch, deployment_location: str | None
) -> None:
    """The route resolves the location without the call's optional params, which Logging reads, so the two must
    agree once the resolved call is logged: the optional params Logging then holds are the Messages body fields."""
    from litellm.litellm_core_utils.litellm_logging import resolve_vertex_location_for_cost

    monkeypatch.setattr(litellm, "vertex_location", "europe-west4")
    deployment: Final = {} if deployment_location is None else {"vertex_location": deployment_location}
    logger: Final = _priced_logger(deployment)
    request: Final = {**_CALL, **deployment, "model": "vertex_ai/claude-opus-5-5", "litellm_logging_obj": logger}

    priced_at: Final = messages_host.settings({"litellm_logging_obj": logger}, request)["pricing"]
    messages_host.log_resolved_call(request, "vertex_ai")
    logged_at: Final = resolve_vertex_location_for_cost(
        custom_llm_provider="vertex_ai",
        litellm_params=logger.litellm_params,
        optional_params=logger.optional_params,
        model=logger.model,
    )

    assert isinstance(priced_at, dict)
    assert priced_at["vertex_location"] == logged_at == (deployment_location or "europe-west4")


def test_settings_leave_pricing_to_python_for_an_invoked_bedrock_model_id() -> None:
    logger: Final = _priced_logger(_ROUTED)

    settings: Final = messages_host.settings(
        {"litellm_logging_obj": logger}, {**_CALL, "model": "anthropic/claude-opus-5-5", "model_id": "arn:profile"}
    )

    assert settings["pricing"] is None
