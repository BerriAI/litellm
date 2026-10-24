import asyncio
import errno
import os
import threading
import time
from hashlib import sha256
from pathlib import Path
from typing import Final

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.tokenize import TokenizeResponse, validate_tokenize_backend
from tests.test_litellm_rust.support.isolation import rebound
from tests.test_litellm_rust.support.recording_server import (
    RecordingServer,
    ResponseSpec,
)
from tests.test_litellm_rust.support.routing import assert_served_natively, call_public

pytestmark = pytest.mark.requires_rust_extension


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("prefix", ["hosted_vllm", "vllm"])
@pytest.mark.parametrize(
    ("include", "token_ids", "tokens"),
    [
        ((), None, None),
        (("token_ids",), [101, 102], None),
        (("tokens",), None, ["Hello", " world"]),
        (("token_ids", "tokens"), [101, 102], ["Hello", " world"]),
    ],
    ids=["count-only", "token-ids", "tokens", "both"],
)
async def test_vllm_returns_only_requested_representations(
    server: RecordingServer,
    asynchronous: bool,
    prefix: str,
    include: tuple[str, ...],
    token_ids: list[int] | None,
    tokens: list[str] | None,
) -> None:
    response: Final = await call_public(
        litellm.tokenize,
        litellm.atokenize,
        {
            "model": f"{prefix}/acme/model",
            "prompt": "Hello world",
            "include": include,
            "user": "opaque-user",
        },
        server,
        asynchronous=asynchronous,
        metadata={"opaque": object()},
    )
    assert isinstance(response, TokenizeResponse)
    assert response.model == "acme/model"
    assert response.count == 2
    assert response.token_ids == token_ids
    assert response.tokens == tokens
    assert_served_natively(server, response)
    sent: Final = server.requests[0]
    assert sent.path == "/tokenize"
    assert sent.body["model"] == "acme/model"
    assert sent.body["prompt"] == "Hello world"
    assert sent.headers["authorization"] == "Bearer test-key"


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    ("include", "tokens"),
    [((), None), (("tokens",), ["Hello", " world"])],
    ids=["count-only", "tokens"],
)
async def test_watsonx_returns_only_requested_token_text(
    server: RecordingServer,
    asynchronous: bool,
    include: tuple[str, ...],
    tokens: list[str] | None,
) -> None:
    server.default_response = ResponseSpec(body={"result": {"token_count": 2, "tokens": ["Hello", " world"]}})
    response: Final = await call_public(
        litellm.tokenize,
        litellm.atokenize,
        {
            "model": "watsonx/ibm/granite-3-8b-instruct",
            "prompt": "Hello world",
            "include": include,
            "project_id": "project",
            "api_version": "2024-01-01",
            "token": "bearer-token",
        },
        server,
        asynchronous=asynchronous,
    )
    assert isinstance(response, TokenizeResponse)
    assert response.model == "ibm/granite-3-8b-instruct"
    assert response.count == 2
    assert response.token_ids is None
    assert response.tokens == tokens
    assert_served_natively(server, response)
    sent: Final = server.requests[0]
    assert sent.path.startswith("/ml/v1/text/tokenization?")
    assert "version=2024-01-01" in sent.path
    assert sent.headers["authorization"] == "Bearer bearer-token"
    assert sent.body["input"] == "Hello world"
    assert sent.body["model_id"] == "ibm/granite-3-8b-instruct"
    assert sent.body["project_id"] == "project"


async def test_deployment_hook_rewrites_input_before_native_projection(
    server: RecordingServer,
) -> None:
    class Rewrite(CustomLogger):
        async def async_pre_call_deployment_hook(
            self, kwargs: dict[str, object], call_type: object
        ) -> dict[str, object]:
            assert call_type == "atokenize"
            return {**kwargs, "prompt": "rewritten"}

    with rebound(litellm, "callbacks", [Rewrite()]):
        response: Final = await litellm.atokenize(
            model="hosted_vllm/acme/model",
            prompt="original",
            api_base=server.base_url,
            api_key="test-key",
        )
    assert_served_natively(server, response)
    assert server.requests[0].body["prompt"] == "rewritten"


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    ("include", "token_ids"),
    [((), None), (("token_ids",), [101, 102])],
    ids=["count-only", "token-ids"],
)
async def test_router_resolves_alias_before_tokenization(
    server: RecordingServer,
    asynchronous: bool,
    include: tuple[str, ...],
    token_ids: list[int] | None,
) -> None:
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "tokenizer",
                "litellm_params": {
                    "model": "hosted_vllm/acme/model",
                    "api_base": server.base_url,
                    "api_key": "deployment-key",
                    "temperature": 0.7,
                    "max_tokens": 128,
                    "add_special_tokens": False,
                },
            }
        ]
    )
    response: Final = (
        await router.atokenize(model="tokenizer", prompt="Hello world", include=include)
        if asynchronous
        else await asyncio.to_thread(router.tokenize, model="tokenizer", prompt="Hello world", include=include)
    )
    assert isinstance(response, TokenizeResponse)
    assert response.model == "acme/model"
    assert response.token_ids == token_ids
    assert response.tokens is None
    assert_served_natively(server, response)
    assert server.requests[0].body["model"] == "acme/model"
    assert server.requests[0].headers["authorization"] == "Bearer deployment-key"

    assert server.requests[0].body["add_special_tokens"] is False
    assert "temperature" not in server.requests[0].body
    assert "max_tokens" not in server.requests[0].body


def _local_backend() -> dict[str, object]:
    artifact: Final = Path(litellm.__file__).parent / "litellm_core_utils/tokenizers/anthropic_tokenizer.json"
    return {
        "type": "local",
        "config": {
            "id": "anthropic-local",
            "revision": "test-revision",
            "engine": {"type": "hugging_face"},
            "artifact": {
                "type": "file",
                "path": str(artifact),
                "sha256": sha256(artifact.read_bytes()).hexdigest(),
            },
        },
    }


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    ("include", "has_token_ids"),
    [((), False), (("token_ids",), True)],
    ids=["count-only", "token-ids"],
)
async def test_local_sdk_tokenization_uses_configured_artifact_without_provider_io(
    server: RecordingServer, asynchronous: bool, include: tuple[str, ...], has_token_ids: bool
) -> None:
    server.expected_requests = 0
    kwargs: Final = {
        "model": "local-alias",
        "prompt": "Hello world",
        "include": include,
        "backend": _local_backend(),
    }
    response: Final = (
        await litellm.atokenize(**kwargs) if asynchronous else await asyncio.to_thread(litellm.tokenize, **kwargs)
    )
    assert isinstance(response, TokenizeResponse)
    assert response.model == "anthropic-local"
    assert response.count > 0
    if has_token_ids:
        assert response.token_ids
        assert response.tokens is None
    else:
        assert response.token_ids is None
        assert response.tokens is None
    assert_served_natively(server, response, count=0)
    assert response._hidden_params["x-litellm-rust"] is True


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_local_sdk_can_return_both_representations(
    server: RecordingServer,
    asynchronous: bool,
) -> None:
    server.expected_requests = 0
    kwargs: Final = {
        "model": "local-alias",
        "prompt": "Hello world",
        "include": ("token_ids", "tokens"),
        "backend": _local_backend(),
    }
    response: Final = (
        await litellm.atokenize(**kwargs) if asynchronous else await asyncio.to_thread(litellm.tokenize, **kwargs)
    )
    assert isinstance(response, TokenizeResponse)
    assert response.model == "anthropic-local"
    assert response.token_ids
    assert response.tokens
    assert len(response.token_ids) == len(response.tokens) == response.count
    assert_served_natively(server, response, count=0)


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_typed_backend_is_normalized_before_native_projection(
    server: RecordingServer,
    asynchronous: bool,
) -> None:
    server.expected_requests = 0
    backend: Final = validate_tokenize_backend(_local_backend())
    kwargs: Final = {
        "model": "typed-local-alias",
        "prompt": "Hello world",
        "include": [],
        "backend": backend,
    }
    response: Final = (
        await litellm.atokenize(**kwargs) if asynchronous else await asyncio.to_thread(litellm.tokenize, **kwargs)
    )
    assert isinstance(response, TokenizeResponse)
    assert response.model == "anthropic-local"
    assert response.count > 0
    assert response.token_ids is None
    assert response.tokens is None
    assert_served_natively(server, response, count=0)
    assert response._hidden_params["x-litellm-rust"] is True


@pytest.mark.parametrize(
    "warm_include",
    [(), ("token_ids", "tokens")],
    ids=["count-only", "both"],
)
async def test_async_local_tokenization_yields_during_cold_and_warm_execution(
    tmp_path: Path,
    server: RecordingServer,
    warm_include: tuple[str, ...],
) -> None:
    server.expected_requests = 0
    source_artifact: Final = Path(litellm.__file__).parent / "litellm_core_utils/tokenizers/anthropic_tokenizer.json"
    artifact_content: Final = source_artifact.read_text()
    artifact_path: Final = tmp_path / "tokenizer.json"
    os.mkfifo(artifact_path)
    reader_connected: Final = threading.Event()
    progress: Final = threading.Event()
    progressed_before_release: Final = threading.Event()
    watchdog_released: Final = threading.Event()

    def feed_artifact() -> None:
        deadline: Final = time.monotonic() + 5

        def connect_reader() -> int | None:
            while True:
                try:
                    return os.open(artifact_path, os.O_WRONLY | os.O_NONBLOCK)
                except OSError as error:
                    if error.errno != errno.ENXIO:
                        raise
                    if time.monotonic() >= deadline:
                        watchdog_released.set()
                        return None
                    time.sleep(0.01)

        descriptor: Final = connect_reader()
        if descriptor is None:
            return
        os.set_blocking(descriptor, True)
        with os.fdopen(descriptor, "wb") as artifact:
            reader_connected.set()
            if progress.wait(timeout=5):
                progressed_before_release.set()
            else:
                watchdog_released.set()
            artifact.write(artifact_content.encode())

    writer: Final = threading.Thread(target=feed_artifact, daemon=True)
    writer.start()
    backend_id: Final = "async-cold-warm-both" if warm_include else "async-cold-warm-count-only"
    backend: Final = {
        "type": "local",
        "config": {
            "id": backend_id,
            "revision": "test-revision",
            "engine": {"type": "hugging_face"},
            "artifact": {
                "type": "file",
                "path": str(artifact_path),
                "sha256": sha256(artifact_content.encode()).hexdigest(),
            },
        },
    }

    async def signal_progress() -> None:
        async with asyncio.timeout(5):
            while not reader_connected.is_set():
                await asyncio.sleep(0)
        progress.set()

    cold_call: Final = asyncio.create_task(
        litellm.atokenize(model=backend_id, prompt="Hello world", include=[], backend=backend)
    )
    signal: Final = asyncio.create_task(signal_progress())
    try:
        cold_response: Final = await cold_call
        await signal
    finally:
        if not signal.done():
            signal.cancel()
        await asyncio.gather(signal, return_exceptions=True)
        progress.set()
        writer.join(timeout=5)
    assert not writer.is_alive()
    assert reader_connected.is_set()
    assert not watchdog_released.is_set()
    assert progressed_before_release.is_set()
    assert cold_response.count > 0
    assert cold_response.token_ids is None
    assert cold_response.tokens is None
    assert_served_natively(server, cold_response, count=0)

    artifact_path.unlink()
    warm_ticks: Final = asyncio.Queue[float]()
    warm_started: Final = asyncio.Queue[float]()
    warm_completed: Final = asyncio.Queue[float]()
    heartbeat_stopped: Final = asyncio.Event()
    loop: Final = asyncio.get_running_loop()
    measurement_started_at: Final = loop.time()

    async def heartbeat() -> None:
        while not heartbeat_stopped.is_set():
            await asyncio.sleep(0.01)
            if heartbeat_stopped.is_set():
                break
            await warm_ticks.put(loop.time())

    async def tokenize_warm() -> TokenizeResponse:
        await warm_started.put(loop.time())
        response: Final = await litellm.atokenize(
            model=backend_id,
            prompt="Hello world " * 200_000,
            include=warm_include,
            backend=backend,
        )
        await warm_completed.put(loop.time())
        return response

    heartbeat_task: Final = asyncio.create_task(heartbeat())
    warm_call: Final = asyncio.create_task(tokenize_warm())
    try:
        warm_response: Final = await warm_call
    finally:
        heartbeat_stopped.set()
        await heartbeat_task

    start_time: Final = warm_started.get_nowait()
    completion_time: Final = warm_completed.get_nowait()
    tick_count: Final = warm_ticks.qsize()
    tick_times: Final = tuple(warm_ticks.get_nowait() for _ in range(tick_count))
    assert tick_times
    assert any(start_time < tick_time < completion_time for tick_time in tick_times)
    first_tick_latency: Final = tick_times[0] - measurement_started_at
    inter_tick_gaps: Final = tuple(later - earlier for earlier, later in zip(tick_times, tick_times[1:]))
    assert first_tick_latency < 0.5
    assert all(gap < 0.5 for gap in inter_tick_gaps)
    assert warm_response.count > cold_response.count
    if warm_include:
        assert warm_response.token_ids
        assert warm_response.tokens
        assert len(warm_response.token_ids) == len(warm_response.tokens) == warm_response.count
    else:
        assert warm_response.token_ids is None
        assert warm_response.tokens is None
    assert_served_natively(server, warm_response, count=0)


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    ("include", "has_token_ids"),
    [((), False), (("token_ids",), True)],
    ids=["count-only", "token-ids"],
)
async def test_local_router_returns_configured_tokenizer_identity(
    server: RecordingServer, asynchronous: bool, include: tuple[str, ...], has_token_ids: bool
) -> None:
    server.expected_requests = 0
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "local-alias",
                "litellm_params": {
                    "model": "local-alias",
                    "backend": _local_backend(),
                },
            }
        ]
    )
    response: Final = (
        await router.atokenize(model="local-alias", prompt="Hello world", include=include)
        if asynchronous
        else await asyncio.to_thread(router.tokenize, model="local-alias", prompt="Hello world", include=include)
    )
    assert isinstance(response, TokenizeResponse)
    assert response.model == "anthropic-local"
    assert response.count > 0
    if has_token_ids:
        assert response.token_ids
        assert response.tokens is None
    else:
        assert response.token_ids is None
        assert response.tokens is None
    assert_served_natively(server, response, count=0)
    assert response._hidden_params["x-litellm-rust"] is True


async def test_caller_temperature_is_rejected_before_provider_io(server: RecordingServer) -> None:
    server.expected_requests = 0
    with pytest.raises(litellm.BadRequestError):
        await litellm.atokenize(
            model="hosted_vllm/acme/model",
            prompt="Hello",
            api_base=server.base_url,
            api_key="test-key",
            temperature=0.7,
            num_retries=0,
        )
    assert_served_natively(server, count=0)


@pytest.mark.parametrize(
    "include",
    [None, ["unknown"], ["tokens", "tokens"]],
    ids=["null", "unknown-entry", "duplicate"],
)
async def test_invalid_include_is_rejected_before_provider_io(
    server: RecordingServer,
    include: object,
) -> None:
    server.expected_requests = 0
    with pytest.raises(litellm.BadRequestError):
        await litellm.atokenize(
            model="hosted_vllm/acme/model",
            prompt="Hello",
            api_base=server.base_url,
            api_key="test-key",
            include=include,
        )
    assert_served_natively(server, count=0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("output", "count"),
        ("messages", []),
        ("tools", []),
        ("image", "encoded"),
        ("images", []),
        ("chat_template", "template"),
        ("add_generation_prompt", True),
    ],
)
async def test_unsupported_input_fields_are_rejected_before_provider_io(
    server: RecordingServer,
    field: str,
    value: object,
) -> None:
    server.expected_requests = 0
    with pytest.raises(litellm.BadRequestError):
        await litellm.atokenize(
            model="hosted_vllm/acme/model",
            prompt="Hello",
            api_base=server.base_url,
            api_key="test-key",
            **{field: value},
        )
    assert_served_natively(server, count=0)


async def test_unknown_provider_option_is_rejected_before_provider_io(server: RecordingServer) -> None:
    server.expected_requests = 0
    with pytest.raises(litellm.BadRequestError):
        await litellm.atokenize(
            model="hosted_vllm/acme/model",
            prompt="Hello",
            api_base=server.base_url,
            api_key="test-key",
            unsupported_option=True,
        )
    assert_served_natively(server, count=0)


@pytest.mark.parametrize("add_special_tokens", [True, False], ids=["enabled", "disabled"])
async def test_vllm_honors_explicit_special_token_policy(
    server: RecordingServer,
    add_special_tokens: bool,
) -> None:
    response: Final = await litellm.atokenize(
        model="hosted_vllm/acme/model",
        prompt="Hello",
        api_base=server.base_url,
        api_key="test-key",
        include=["token_ids"],
        add_special_tokens=add_special_tokens,
    )
    assert isinstance(response, TokenizeResponse)
    assert response.model == "acme/model"
    assert server.requests[0].body["add_special_tokens"] is add_special_tokens
    assert_served_natively(server, response)


async def test_watsonx_rejects_special_token_option_before_provider_io(server: RecordingServer) -> None:
    server.expected_requests = 0
    with pytest.raises(litellm.BadRequestError):
        await litellm.atokenize(
            model="watsonx/ibm/granite-3-8b-instruct",
            prompt="Hello",
            api_base=server.base_url,
            api_key="test-key",
            include=["tokens"],
            add_special_tokens=True,
        )
    assert_served_natively(server, count=0)


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_router_uses_trusted_watsonx_deployment_options(
    server: RecordingServer,
    asynchronous: bool,
) -> None:
    server.default_response = ResponseSpec(body={"result": {"token_count": 2, "tokens": ["Hello", " world"]}})
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "watsonx-tokenizer",
                "litellm_params": {
                    "model": "watsonx/ibm/granite-3-8b-instruct",
                    "api_base": server.base_url,
                    "token": "deployment-token",
                    "project_id": "deployment-project",
                    "api_version": "2024-03-13",
                },
            }
        ]
    )
    response: Final = (
        await router.atokenize(model="watsonx-tokenizer", prompt="Hello world", include=["tokens"])
        if asynchronous
        else await asyncio.to_thread(
            router.tokenize, model="watsonx-tokenizer", prompt="Hello world", include=["tokens"]
        )
    )
    assert isinstance(response, TokenizeResponse)
    assert response.model == "ibm/granite-3-8b-instruct"
    assert response.count == 2
    assert response.tokens == ["Hello", " world"]
    assert_served_natively(server, response)
    sent: Final = server.requests[0]
    assert "version=2024-03-13" in sent.path
    assert sent.headers["authorization"] == "Bearer deployment-token"
    assert sent.body["project_id"] == "deployment-project"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("project_id", "caller-project"),
        ("api_version", "2025-01-01"),
        ("_tokenize_deployment_params", {"temperature": 0.1}),
    ],
)
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_router_rejects_request_sourced_watsonx_fields_before_provider_io(
    server: RecordingServer,
    field: str,
    value: object,
    asynchronous: bool,
) -> None:
    server.expected_requests = 0
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "watsonx-tokenizer",
                "litellm_params": {
                    "model": "watsonx/ibm/granite-3-8b-instruct",
                    "api_base": server.base_url,
                    "api_key": "deployment-key",
                },
            }
        ]
    )
    kwargs: Final = {"model": "watsonx-tokenizer", "prompt": "Hello", field: value}
    expected_error: Final = ValueError if field.startswith("_") else litellm.BadRequestError

    async def invoke_router() -> TokenizeResponse:
        if asynchronous:
            return await router.atokenize(**kwargs)
        return await asyncio.to_thread(router.tokenize, **kwargs)

    with pytest.raises(expected_error):
        await invoke_router()
    assert_served_natively(server, count=0)


async def test_default_include_returns_count_only(server: RecordingServer) -> None:
    response: Final = await litellm.atokenize(
        model="hosted_vllm/acme/model",
        prompt="Hello",
        api_base=server.base_url,
        api_key="test-key",
    )
    assert isinstance(response, TokenizeResponse)
    assert response.model == "acme/model"
    assert response.count == 2
    assert response.token_ids is None
    assert response.tokens is None
    assert_served_natively(server, response)


async def test_empty_prompt_is_forwarded_without_coercion(server: RecordingServer) -> None:
    server.default_response = ResponseSpec(body={"count": 0, "tokens": [], "token_strs": []})
    response: Final = await litellm.atokenize(
        model="hosted_vllm/acme/model",
        prompt="",
        include=[],
        api_base=server.base_url,
        api_key="test-key",
    )
    assert isinstance(response, TokenizeResponse)
    assert response.model == "acme/model"
    assert response.count == 0
    assert response.token_ids is None
    assert response.tokens is None
    assert server.requests[0].body["prompt"] == ""
    assert_served_natively(server, response)


async def test_upstream_status_remains_a_public_error(server: RecordingServer) -> None:
    server.default_response = ResponseSpec(
        status=429,
        body={"error": "rate limited"},
        headers={
            "Retry-After": "31",
            "x-request-id": "tokenize-rate-limit",
            "request-id": "tokenize-request",
            "x-correlation-id": "tokenize-correlation",
            "x-global-transaction-id": "tokenize-global-transaction",
            "traceparent": "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01",
            "Authorization": "Bearer unsafe-response-value",
            "Set-Cookie": "unsafe=header",
        },
    )
    with pytest.raises(litellm.RateLimitError) as caught:
        await litellm.atokenize(
            model="hosted_vllm/acme/model",
            prompt="Hello",
            api_base=server.base_url,
            api_key="test-key",
            num_retries=0,
        )
    assert caught.value.status_code == 429
    assert caught.value.headers["retry-after"] == "31"
    assert caught.value.headers["x-request-id"] == "tokenize-rate-limit"
    assert caught.value.headers["request-id"] == "tokenize-request"
    assert caught.value.headers["x-correlation-id"] == "tokenize-correlation"
    assert caught.value.headers["x-global-transaction-id"] == "tokenize-global-transaction"
    assert caught.value.headers["traceparent"] == "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"
    assert "authorization" not in caught.value.headers
    assert "set-cookie" not in caught.value.headers
    assert_served_natively(server)
