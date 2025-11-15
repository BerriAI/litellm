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

