from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Final, cast

import httpx

from tests.test_litellm._fixture_recorder import (
    ProviderSpec,
    pending_requests,
    record_case,
)
from tests.test_litellm._json_fs_cache import JsonFileCache


def _request_value(model: str = "test/model") -> dict[str, object]:
    return {
        "request": {
            "provider": "test-provider",
            "sdk_kwargs": {"model": model},
            "provider_request": {"method": "POST", "path": "/v1/test", "body": {"model": model}},
        }
    }


def _sdk_call(**kwargs: object) -> None:
    api_base: Final = kwargs["api_base"]
    model: Final = kwargs["model"]
    assert isinstance(api_base, str)
    assert isinstance(model, str)
    response: Final = httpx.post(
        f"{api_base}/v1/test",
        content=json.dumps({"model": model}).encode(),
        headers={"content-type": "application/json"},
    )
    response.raise_for_status()


def _same_wire_sdk_call(**kwargs: object) -> None:
    api_base: Final = kwargs["api_base"]
    assert isinstance(api_base, str)
    response: Final = httpx.post(
        f"{api_base}/v1/test",
        content=b'{"constant":true}',
        headers={"content-type": "application/json"},
    )
    response.raise_for_status()


def test_request_only_recording_persists_pending_fixture_without_calling_upstream(tmp_path: Path) -> None:
    spec: Final = ProviderSpec(
        name="test-provider",
        model="test/model",
        upstream_base="http://127.0.0.1:1",
        api_key="test-key",
    )
    sdk_kwargs: Final[dict[str, object]] = {"model": spec.model}

    result: Final = record_case(
        spec,
        tmp_path,
        sdk_kwargs,
        requests_only=True,
        sdk_call=cast(Callable[..., object], _sdk_call),
    )
    values: Final = JsonFileCache(tmp_path / spec.name).values()

    assert result.response is None
    assert len(values) == 1
    assert values[0] == {"request": result.request.model_dump(mode="json")}


def test_pending_requests_excludes_completed_fixtures(tmp_path: Path) -> None:
    cache: Final = JsonFileCache(tmp_path)
    pending: Final = _request_value()
    response: Final[dict[str, object]] = {"status_code": 200, "headers": {}, "body": {}}
    completed: Final[dict[str, object]] = {
        **_request_value("test/completed"),
        "response": response,
    }
    cache.put({"case": "pending"}, pending)
    cache.put({"case": "completed"}, completed)

    requests: Final = pending_requests(cache)

    assert len(requests) == 1
    assert requests[0].sdk_kwargs["model"] == "test/model"


def test_request_cache_distinguishes_sdk_inputs_with_identical_wire_requests(tmp_path: Path) -> None:
    spec: Final = ProviderSpec(
        name="test-provider",
        model="test/model-a",
        upstream_base="http://127.0.0.1:1",
        api_key="test-key",
    )
    sdk_call: Final = cast(Callable[..., object], _same_wire_sdk_call)

    first: Final = record_case(spec, tmp_path, {"model": "test/model-a"}, True, sdk_call)
    second: Final = record_case(spec, tmp_path, {"model": "test/model-b"}, True, sdk_call)

    assert not first.cache_hit
    assert not second.cache_hit
    assert len(JsonFileCache(tmp_path / spec.name).values()) == 2
