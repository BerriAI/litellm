import os
import sys

sys.path.insert(
    0, os.path.abspath("../../..")
)

from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.proxy.public_endpoints import router
from litellm.types.utils import LlmProviders


def test_get_supported_providers_returns_enum_values():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    response = client.get("/public/providers")

    assert response.status_code == 200
    expected_providers = sorted(provider.value for provider in LlmProviders)
    assert response.json() == expected_providers


def test_get_provider_fields_returns_metadata():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    response = client.get("/public/providers/fields")

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)

    provider_lookup = {item["provider"]: item for item in payload}
    assert "OpenAI" in provider_lookup

    openai_fields = provider_lookup["OpenAI"]
    assert openai_fields["provider_display_name"] == "OpenAI"
    assert openai_fields["litellm_provider"] == "openai"

    credential_keys = {field["key"] for field in openai_fields["credential_fields"]}
    assert {"api_base", "api_key"}.issubset(credential_keys)

    # Every provider exposed by `/public/providers` (i.e. every LlmProviders value)
    # should have a corresponding entry in `/public/providers/fields`.
    expected_litellm_providers = {provider.value for provider in LlmProviders}
    actual_litellm_providers = {item["litellm_provider"] for item in payload}
    assert expected_litellm_providers.issubset(actual_litellm_providers)

    # Sanity check for runwayml specifically – it should be present and use the
    # default API base + API key credential fields at minimum.
    runway_entries = [
        item for item in payload if item["litellm_provider"] == "runwayml"
    ]
    assert (
        len(runway_entries) >= 1
    ), "Expected runwayml provider metadata in /public/providers/fields"
    runway_credential_keys = {
        field["key"] for field in runway_entries[0]["credential_fields"]
    }
    assert {"api_base", "api_key"}.issubset(runway_credential_keys)


def test_get_model_provider_map_returns_correct_structure():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    response = client.get("/public/model_provider_map")

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, dict)

    # Verify structure: each entry should have litellm_provider, optionally mode
    for model_name, model_info in payload.items():
        assert isinstance(model_name, str)
        assert isinstance(model_info, dict)
        assert "litellm_provider" in model_info
        assert isinstance(model_info["litellm_provider"], str)
        assert len(model_info["litellm_provider"]) > 0
        
        # If mode exists, it should be a valid string
        if "mode" in model_info:
            assert isinstance(model_info["mode"], str)
            assert len(model_info["mode"]) > 0

    # Verify some common models exist (if model_cost is populated)
    if len(payload) > 0:
        # Check for at least one OpenAI model
        openai_models = [
            model for model, info in payload.items()
            if info.get("litellm_provider") == "openai"
        ]
        # If OpenAI models exist, verify structure
        if openai_models:
            sample_model = openai_models[0]
            assert "litellm_provider" in payload[sample_model]
            assert payload[sample_model]["litellm_provider"] == "openai"
            # Most OpenAI models should have mode="chat"
            if "mode" in payload[sample_model]:
                assert payload[sample_model]["mode"] in [
                    "chat", "completion", "embedding", 
                    "image_generation", "audio_transcription"
                ]

