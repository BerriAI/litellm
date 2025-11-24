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
    app_instance = FastAPI()
    app_instance.include_router(router)
    client = TestClient(app_instance)

    response = client.get("/public/providers")

    assert response.status_code == 200
    expected_providers = sorted(provider.value for provider in LlmProviders)
    assert response.json() == expected_providers


def test_get_provider_create_fields():
    app_instance = FastAPI()
    app_instance.include_router(router)
    client = TestClient(app_instance)

    response = client.get("/public/providers/fields")

    assert response.status_code == 200

    response_data = response.json()

    assert isinstance(response_data, list)

    assert len(response_data) > 0

    first_provider = response_data[0]
    assert "provider" in first_provider
    assert "provider_display_name" in first_provider
    assert "litellm_provider" in first_provider
    assert "credential_fields" in first_provider

    assert isinstance(first_provider["credential_fields"], list)

    has_detailed_fields = any(
        provider.get("credential_fields") and len(provider.get("credential_fields", [])) > 0
        for provider in response_data
    )
    assert has_detailed_fields, "Expected at least one provider to have detailed credential fields"

