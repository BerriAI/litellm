from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from typing import Final

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

import litellm
import openai
from litellm import Router

_KEY: Final = "synthetic-family-key"
_PROMPT: Final = "family wire probe"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_OVERFLOW_TEXT: Final = "This model's maximum context length is 4096 tokens. However, you requested 9000 tokens."
_INFERENCEAPIS_TEXT: Final = "Input too long: 150055 input tokens, limit is 131072 for this model"
_PLAIN_TEXT: Final = "Invalid value for 'temperature': expected a number between 0 and 2."
_FIREWORKS_PARAM_TEXT: Final = (
    "Input should be 'error' or 'truncate', field: 'context_length_exceeded_behavior', value: 'foo'"
)
_TOKEN_REPLY: Final = Reply(
    body=json.dumps(
        {"access_token": "synthetic-token", "expires_at": 4102444800000, "expires_in": 3600, "token_type": "bearer"}
    ).encode()
)
_STATUS_MAPPED_JSON_PROVIDERS: Final = (
    "veniceai",
    "xiaomi_mimo",
    "scaleway",
    "abliteration",
    "llamagate",
    "gmi",
    "sarvam",
    "assemblyai",
    "charity_engine",
    "aihubmix",
    "crusoe",
    "neosantara",
    "empiriolabs",
)
_EXTRA_BODY_MEMBERS: Final = (*_STATUS_MAPPED_JSON_PROVIDERS, "inferenceapis")
_OPENAI_COMPATIBLE_JSON_MEMBER: Final = "nano-gpt"
_PLAIN_BUILDERS: Final = (
    *_STATUS_MAPPED_JSON_PROVIDERS,
    "a2a",
    "amazon_nova",
    "cloudflare",
    "compactifai",
    "custom",
    "datarobot",
    "heroku",
    "langflow",
    "langgraph",
    "maritalk",
    "minimax",
    "nadir",
    "ollama",
    "ollama_chat",
    "oobabooga",
    "ovhcloud",
    "petals",
    "snowflake",
    "text-completion-codestral",
    "text-completion-inception",
)
_HANDLER_BRANCHES: Final = ("openai", "anthropic", "deepinfra", "gemini")


def _oci_key() -> str:
    key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    ).decode()


def _plain_call(provider: str) -> Callable[[str], dict[str, JsonValue]]:
    return lambda url: {"model": f"{provider}/m", "api_base": url, "api_key": _KEY}


_SPECIAL_BUILDERS: Final[Mapping[str, Callable[[str], dict[str, JsonValue]]]] = {
    "triton": lambda url: {"model": "triton/m", "api_base": f"{url}/v2/models/m/generate"},
    "watsonx_text": lambda url: {"model": "watsonx_text/m", "api_base": url, "token": "t", "project_id": "p"},
    "gigachat": lambda url: {
        "model": "gigachat/m",
        "api_base": url,
        "api_key": _KEY,
        "gigachat_auth_url": f"{url}/oauth",
    },
    "scaledown": lambda url: {"model": "scaledown/summarize", "api_base": url, "api_key": _KEY},
    "gradient_ai": lambda url: {"model": "gradient_ai/m", "api_base": url, "api_key": _KEY, "drop_params": True},
    "oci": lambda url: {
        "model": "oci/meta.llama-3.3-70b-instruct",
        "api_base": url,
        "oci_key": _oci_key(),
        "oci_user": "ocid1.user.oc1..synthetic",
        "oci_fingerprint": "aa:bb:cc",
        "oci_tenancy": "ocid1.tenancy.oc1..synthetic",
        "oci_region": "us-chicago-1",
        "oci_compartment_id": "ocid1.compartment.oc1..synthetic",
    },
}
_BUILDER_CALLS: Final[Mapping[str, Callable[[str], dict[str, JsonValue]]]] = {
    **{provider: _plain_call(provider) for provider in _PLAIN_BUILDERS},
    **_SPECIAL_BUILDERS,
}


def _error(status: int, message: str, code: str | None = None) -> Reply:
    error: Final[dict[str, JsonValue]] = {"message": message, "type": "invalid_request_error", "code": code}
    return Reply(status=status, body=json.dumps({"error": error}).encode())


def _is_token_request(request: Request) -> bool:
    return "oauth" in request.target or "token" in request.target


def _scripted(reply: Reply) -> Callable[[Request], Reply]:
    return lambda request: _TOKEN_REPLY if _is_token_request(request) else reply


def _prompted(wire: Wire) -> int:
    return sum(1 for request in wire.drain() if not _is_token_request(request))


def _completion(model: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": "chatcmpl-family",
                "object": "chat.completion",
                "created": 1,
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            }
        ).encode()
    )


def _answer(response: object) -> str | None:
    assert isinstance(response, litellm.ModelResponse)
    choice: Final = response.choices[0]
    assert isinstance(choice, litellm.Choices)
    return choice.message.content


async def _raised(call: dict[str, JsonValue]) -> BaseException:
    with pytest.raises(openai.APIError) as caught:
        await litellm.acompletion(messages=[{"role": "user", "content": _PROMPT}], num_retries=0, **call)
    return caught.value


@pytest.mark.parametrize(
    ("text", "code"),
    ((_OVERFLOW_TEXT, "context_length_exceeded"), (_INFERENCEAPIS_TEXT, "upstream_rejected")),
    ids=("standard-phrase", "inference-apis-phrase"),
)
def test_an_overflow_400_from_inference_apis_raises_context_window_exceeded(text: str, code: str) -> None:
    with wire_server(_scripted(_error(400, text, code))) as wire:
        raised: Final = asyncio.run(_raised(_plain_call("inferenceapis")(wire.url)))
        assert isinstance(raised, litellm.ContextWindowExceededError), repr(raised)
        assert raised.status_code == 400
        assert text in str(raised)
        assert _prompted(wire) == 1


@pytest.mark.parametrize("text", (_OVERFLOW_TEXT, _PLAIN_TEXT), ids=("overflow", "plain"))
@pytest.mark.parametrize("provider", sorted(_BUILDER_CALLS))
def test_a_400_stays_a_bad_request_for_every_provider_without_its_own_mapper(provider: str, text: str) -> None:
    with wire_server(_scripted(_error(400, text))) as wire:
        raised: Final = asyncio.run(_raised(_BUILDER_CALLS[provider](wire.url)))
        assert isinstance(raised, litellm.BadRequestError), repr(raised)
        assert not isinstance(raised, litellm.ContextWindowExceededError), repr(raised)
        assert text in str(raised)
        assert _prompted(wire) == 1


@pytest.mark.parametrize("provider", _HANDLER_BRANCHES)
def test_the_inference_apis_overflow_phrase_raises_context_window_exceeded_on_providers_with_a_mapper(
    provider: str,
) -> None:
    with wire_server(_scripted(_error(400, _INFERENCEAPIS_TEXT, "upstream_rejected"))) as wire:
        raised: Final = asyncio.run(_raised(_plain_call(provider)(wire.url)))
        assert isinstance(raised, litellm.ContextWindowExceededError), repr(raised)
        assert _INFERENCEAPIS_TEXT in str(raised)
        assert _prompted(wire) == 1


def test_a_param_error_naming_context_length_exceeded_behavior_stays_a_bad_request() -> None:
    with wire_server(_scripted(_error(400, _FIREWORKS_PARAM_TEXT))) as wire:
        call: Final[dict[str, JsonValue]] = {
            "model": "fireworks_ai/accounts/fireworks/models/m",
            "api_base": wire.url,
            "api_key": _KEY,
        }
        raised: Final = asyncio.run(_raised(call))
        assert isinstance(raised, litellm.BadRequestError), repr(raised)
        assert not isinstance(raised, litellm.ContextWindowExceededError), repr(raised)
        assert _prompted(wire) == 1


def test_aiohttp_openai_still_reports_an_upstream_400_as_a_server_error() -> None:
    with wire_server(_scripted(_error(400, _OVERFLOW_TEXT, "context_length_exceeded"))) as wire:
        raised: Final = asyncio.run(_raised(_plain_call("aiohttp_openai")(wire.url)))
        assert isinstance(raised, litellm.InternalServerError), repr(raised)
        assert _prompted(wire) == 1


_FALLBACK_SETUPS: Final[Mapping[str, tuple[str, list[dict[str, list[str]]], list[dict[str, list[str]]]]]] = {
    "context-window-fallbacks": ("inferenceapis/m", [{"primary": ["backup"]}], []),
    "generic-fallbacks-beside-context-window-fallbacks": (
        "xiaomi_mimo/m",
        [{"other": ["backup"]}],
        [{"primary": ["backup"]}],
    ),
}


def _fallback_router(setup: str, primary_url: str, backup_url: str) -> Router:
    primary_model, context_window_fallbacks, fallbacks = _FALLBACK_SETUPS[setup]
    return Router(
        model_list=[
            {
                "model_name": "primary",
                "litellm_params": {"model": primary_model, "api_base": primary_url, "api_key": _KEY},
            },
            {
                "model_name": "other",
                "litellm_params": {"model": "openai/other", "api_base": backup_url, "api_key": _KEY},
            },
            {
                "model_name": "backup",
                "litellm_params": {"model": "openai/backup", "api_base": backup_url, "api_key": _KEY},
            },
        ],
        context_window_fallbacks=context_window_fallbacks,
        fallbacks=fallbacks,
        num_retries=0,
    )


@pytest.mark.parametrize("client", ("sync", "async"))
@pytest.mark.parametrize("setup", sorted(_FALLBACK_SETUPS))
def test_the_configured_fallback_answers_an_overflow(setup: str, client: str) -> None:
    with (
        wire_server(_scripted(_error(400, _OVERFLOW_TEXT, "context_length_exceeded"))) as primary,
        wire_server(lambda _: _completion("backup")) as backup,
    ):
        router: Final = _fallback_router(setup, primary.url, backup.url)
        response: Final[object] = (
            router.completion(model="primary", messages=[{"role": "user", "content": _PROMPT}])
            if client == "sync"
            else asyncio.run(router.acompletion(model="primary", messages=[{"role": "user", "content": _PROMPT}]))
        )
        assert _answer(response) == "ok"
        assert _prompted(primary) == 1
        assert _prompted(backup) == 1


def _forwarding(provider_model: str) -> Callable[[Request], Reply]:
    return lambda _: _completion(provider_model)


@pytest.mark.parametrize("provider", (*_EXTRA_BODY_MEMBERS, _OPENAI_COMPATIBLE_JSON_MEMBER))
def test_every_json_provider_forwards_non_openai_params_in_the_request_body(provider: str) -> None:
    with wire_server(_forwarding("m")) as wire:
        response: Final = asyncio.run(
            litellm.acompletion(
                model=f"{provider}/m",
                api_base=wire.url,
                api_key=_KEY,
                messages=[{"role": "user", "content": _PROMPT}],
                top_k=5,
                chat_template_kwargs={"enable_thinking": False},
                num_retries=0,
            )
        )
        assert _answer(response) == "ok"
        requests: Final = wire.drain()
        assert len(requests) == 1
        body: Final = _JSON_OBJECT.validate_json(requests[0].body)
        assert body["top_k"] == 5
        assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_a_sync_json_provider_call_forwards_non_openai_params_in_the_request_body() -> None:
    with wire_server(_forwarding("m")) as wire:
        litellm.completion(
            model="veniceai/m",
            api_base=wire.url,
            api_key=_KEY,
            messages=[{"role": "user", "content": _PROMPT}],
            top_k=7,
            num_retries=0,
        )
        requests: Final = wire.drain()
        assert len(requests) == 1
        assert _JSON_OBJECT.validate_json(requests[0].body)["top_k"] == 7
