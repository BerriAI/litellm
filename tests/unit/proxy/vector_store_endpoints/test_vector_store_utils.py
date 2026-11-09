import json

import pytest

import litellm
from litellm.proxy.vector_store_endpoints.utils import get_litellm_managed_vector_store
from litellm.vector_stores.vector_store_registry import VectorStoreRegistry


def _registered_store(litellm_params: object) -> dict[str, object]:
    return {
        "vector_store_id": "vs_1",
        "custom_llm_provider": "bedrock",
        "vector_store_name": "kb",
        "vector_store_metadata": '{"source": "db"}',
        "team_id": None,
        "litellm_params": litellm_params,
    }


@pytest.mark.parametrize(
    ("stored_params", "expected_params"),
    [
        (
            '{"api_base": "https://example.com", "embedding": {"api_key": "sk-1", "dims": [1, null]}}',
            {"api_base": "https://example.com", "embedding": {"api_key": "sk-1", "dims": [1, None]}},
        ),
        (
            '{"top_k": 12, "score_threshold": 12.0, "api_version": "12", "stream": true, "verify": false,'
            ' "region": null, "max_tokens": 123456789012345678901234567890}',
            {
                "top_k": 12,
                "score_threshold": 12.0,
                "api_version": "12",
                "stream": True,
                "verify": False,
                "region": None,
                "max_tokens": 123456789012345678901234567890,
            },
        ),
        ('{"z": 1, "a": 2, "a": 3}', {"z": 1, "a": 3}),
        ("{}", {}),
    ],
)
async def test_registry_store_with_json_object_string_params_resolves_to_decoded_params(
    monkeypatch: pytest.MonkeyPatch, stored_params: str, expected_params: dict[str, object]
):
    store = _registered_store(stored_params)
    monkeypatch.setattr(litellm, "vector_store_registry", VectorStoreRegistry([store]))

    resolved = await get_litellm_managed_vector_store("vs_1")

    assert resolved == {**store, "litellm_params": expected_params}
    assert json.dumps(resolved["litellm_params"]) == json.dumps(expected_params)
    assert store["litellm_params"] == stored_params


@pytest.mark.parametrize(
    "stored_params",
    [
        "[1, 2]",
        '[["api_base", "https://example.com"]]',
        '"{\\"api_base\\": \\"https://example.com\\"}"',
        "12",
        "12.5",
        "true",
        "null",
        "{not json",
        "",
    ],
)
async def test_registry_store_with_non_object_or_invalid_json_string_params_resolves_to_empty_params(
    monkeypatch: pytest.MonkeyPatch, stored_params: str
):
    store = _registered_store(stored_params)
    monkeypatch.setattr(litellm, "vector_store_registry", VectorStoreRegistry([store]))

    resolved = await get_litellm_managed_vector_store("vs_1")

    assert resolved == {**store, "litellm_params": {}}
    assert store["litellm_params"] == stored_params


@pytest.mark.parametrize("stored_params", [{"api_base": "https://example.com"}, {}, None])
async def test_registry_store_with_non_string_params_is_returned_as_is(
    monkeypatch: pytest.MonkeyPatch, stored_params: object
):
    store = _registered_store(stored_params)
    monkeypatch.setattr(litellm, "vector_store_registry", VectorStoreRegistry([store]))

    resolved = await get_litellm_managed_vector_store("vs_1")

    assert resolved is store
    assert resolved["litellm_params"] is stored_params
