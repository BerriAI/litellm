from __future__ import annotations

import base64
import os
import uuid
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import JsonValue, TypeAdapter

import litellm
from tests.integration._support.client import JSON_OBJECT, Gateway, object_value
from tests.integration._support.process import OwnedState, graceful_stop_seconds, owned_proxy_process
from tests.integration.caching.response_cache_case import (
    CHAT,
    CHAT_MODEL,
    CHAT_STREAM,
    PROVIDER_KEY,
    Upstream,
    await_entry,
    content_of_first_choice,
    json_response,
    prompt,
    redis_store,
    response_id,
    scripted,
)

IMAGE_COUNT: Final = 8
IMAGE_RANDOM_BYTES: Final = 786_432
SDK_MESSAGES: Final = TypeAdapter(list[ChatCompletionMessageParam])


def _images() -> tuple[str, ...]:
    return tuple(
        "data:image/png;base64," + base64.b64encode(os.urandom(IMAGE_RANDOM_BYTES)).decode() for _ in range(IMAGE_COUNT)
    )


def _one_byte_changed(images: tuple[str, ...], index: int) -> tuple[str, ...]:
    image: Final = images[index]
    middle: Final = len(image) // 2
    replacement: Final = "B" if image[middle] == "A" else "A"
    return (*images[:index], image[:middle] + replacement + image[middle + 1 :], *images[index + 1 :])


def _messages(text: str, images: tuple[str, ...]) -> list[JsonValue]:
    parts: Final[list[JsonValue]] = [
        {"type": "text", "text": text},
        *({"type": "image_url", "image_url": {"url": url}} for url in images),
    ]
    return [{"role": "user", "content": parts}]


def _chat(gateway: Gateway, model: str, messages: list[JsonValue], **extra: JsonValue) -> httpx.Response:
    return gateway.request("POST", "/v1/chat/completions", {"model": model, "messages": messages, **extra})


def _stream(gateway: Gateway, model: str, messages: list[JsonValue]) -> tuple[frozenset[str], str]:
    with (
        OpenAI(base_url=f"{str(gateway.client.base_url).rstrip('/')}/v1", api_key=gateway.key, max_retries=0) as client,
        client.chat.completions.create(
            model=model, messages=SDK_MESSAGES.validate_python(messages), stream=True
        ) as stream,
    ):
        chunks: Final = tuple(stream)
    text: Final = "".join(
        choice.delta.content or "" for chunk in chunks for choice in chunk.choices
    )  # comprehension-ok: flattens chunk choices
    return frozenset(chunk.id for chunk in chunks), text


def test_an_identical_request_with_large_images_is_served_from_the_key_stored_in_redis(gateway: Gateway) -> None:
    upstream: Final = Upstream(gateway.upstream_url)
    store: Final = redis_store()
    messages: Final = _messages(prompt(), _images())
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, json_response(CHAT))
        model: Final = scenario.model(api_base=handle.api_base())
        filled: Final = _chat(gateway, model, messages)
        stored_key: Final = await_entry(store, response_id(filled))
        hit: Final = _chat(gateway, model, messages)
        assert response_id(hit) == response_id(filled)
        assert content_of_first_choice(hit) == "scripted", hit.text
        assert hit.headers.get("x-litellm-cache-key") == stored_key, dict(hit.headers)
        assert upstream.calls(handle.scenario_id) == 1


def test_changing_one_byte_of_one_large_image_reaches_the_upstream_again(gateway: Gateway) -> None:
    upstream: Final = Upstream(gateway.upstream_url)
    store: Final = redis_store()
    text: Final = prompt()
    images: Final = _images()
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, json_response(CHAT))
        model: Final = scenario.model(api_base=handle.api_base())
        filled: Final = _chat(gateway, model, _messages(text, images))
        await_entry(store, response_id(filled))
        changed: Final = _chat(gateway, model, _messages(text, _one_byte_changed(images, IMAGE_COUNT // 2)))
        assert response_id(changed) != response_id(filled), "a different image was served the cached answer"
        assert upstream.calls(handle.scenario_id) == 2


def test_an_identical_stream_with_large_images_is_replayed_from_the_cache(gateway: Gateway) -> None:
    upstream: Final = Upstream(gateway.upstream_url)
    store: Final = redis_store()
    messages: Final = _messages(prompt(), _images())
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, CHAT_STREAM)
        model: Final = scenario.model(api_base=handle.api_base())
        first_ids, first_text = _stream(gateway, model, messages)
        assert first_text == "streamed response"
        await_entry(store, next(iter(first_ids)))
        _, replay_text = _stream(gateway, model, messages)
        assert replay_text == "streamed response"
        assert upstream.calls(handle.scenario_id) == 1


def _modify_params_config(directory: Path, name: str, api_base: str) -> Path:
    rig: Final = JSON_OBJECT.validate_python(
        yaml.safe_load((Path(__file__).resolve().parents[1] / "proxy_config.yaml").read_text())
    )
    litellm_settings: Final = object_value(rig.get("litellm_settings") or {})
    config: Final = {
        **rig,
        "model_list": [
            {
                "model_name": name,
                "litellm_params": {"model": CHAT_MODEL, "api_key": PROVIDER_KEY, "api_base": api_base},
            }
        ],
        "litellm_settings": {**litellm_settings, "modify_params": True},
    }
    path: Final = directory / f"proxy_config-{uuid.uuid4().hex[:8]}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


@pytest.mark.timeout(2 * graceful_stop_seconds() + 120)
def test_a_repeat_request_whose_max_tokens_the_proxy_lowered_is_served_from_the_cache(
    gateway: Gateway, tmp_path: Path
) -> None:
    upstream: Final = Upstream(gateway.upstream_url)
    store: Final = redis_store()
    model_limit: Final = litellm.get_model_info(CHAT_MODEL)["max_output_tokens"]
    assert model_limit is not None
    messages: Final[list[JsonValue]] = [{"role": "user", "content": prompt()}]
    name: Final = f"cache-clamp-{uuid.uuid4().hex[:8]}"
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, json_response(CHAT))
        config: Final = _modify_params_config(tmp_path, name, handle.api_base())
        with owned_proxy_process(
            gateway, tmp_path, {}, config=config, workers=1, state=OwnedState.SHARED_WITH_WORKER
        ) as owned:
            filled: Final = _chat(owned.gateway, name, messages, max_tokens=model_limit * 10)
            forwarded: Final = object_value(upstream.received(handle.scenario_id)[0]["body"])
            forwarded_max_tokens: Final = forwarded.get("max_tokens", forwarded.get("max_completion_tokens"))
            assert isinstance(forwarded_max_tokens, int) and forwarded_max_tokens <= model_limit, forwarded
            await_entry(store, response_id(filled))
            repeat: Final = _chat(owned.gateway, name, messages, max_tokens=model_limit * 10)
            assert response_id(repeat) == response_id(filled), "the repeat request reached the provider again"
            assert upstream.calls(handle.scenario_id) == 1
