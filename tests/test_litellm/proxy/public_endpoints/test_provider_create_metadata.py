import os
import sys
from copy import deepcopy

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

import litellm.proxy.public_endpoints.provider_create_metadata as pcm  # noqa: E402
from litellm.proxy.public_endpoints.provider_create_metadata import (  # noqa: E402
    _normalize_field,
    get_provider_create_metadata,
)


def test_get_provider_create_metadata_includes_openai_fields():
    metadata = get_provider_create_metadata()

    openai_info = next(item for item in metadata if item.provider == "OpenAI")

    assert openai_info.provider_display_name == "OpenAI"
    assert openai_info.litellm_provider == "openai"
    keys = {field.key for field in openai_info.credential_fields}
    assert {"api_base", "api_key"}.issubset(keys)


def test_get_provider_create_metadata_returns_sorted_display_names():
    metadata = get_provider_create_metadata()
    display_names = [item.provider_display_name for item in metadata]

    assert display_names == sorted(display_names, key=str.lower)


def test_get_provider_create_metadata_uses_fallback_fields(monkeypatch):
    overridden_fields = deepcopy(pcm.PROVIDER_CREDENTIAL_FIELDS)
    overridden_fields.pop("Azure", None)
    monkeypatch.setattr(pcm, "PROVIDER_CREDENTIAL_FIELDS", overridden_fields)

    metadata = get_provider_create_metadata()
    azure_info = next(item for item in metadata if item.provider == "Azure")

    fallback_keys = [field.key for field in azure_info.credential_fields]
    assert fallback_keys == ["api_base", "api_key"]
    assert all(field.required is False for field in azure_info.credential_fields)


def test_normalize_field_applies_defaults():
    normalized = _normalize_field({"key": "api_key", "label": "API Key"})

    assert normalized.key == "api_key"
    assert normalized.label == "API Key"
    assert normalized.field_type == "text"
    assert normalized.required is False
    assert normalized.placeholder is None
    assert normalized.options is None
