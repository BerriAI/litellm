from typing import Final

import httpx
from fastapi import FastAPI

from litellm.llms.openai_like.json_loader import JSONProviderRegistry
from litellm.proxy.public_endpoints.public_endpoints import router
from litellm.types.proxy.public_endpoints.public_endpoints import ProviderCreateInfo, SupportedEndpointsResponse
from litellm.utils import get_model_info


async def test_tsubasa_dashboard_fields_resolve_to_a_registered_model() -> None:
    app: Final = FastAPI()
    app.include_router(router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        response: Final = await client.get("/public/providers/fields")
        endpoints_response: Final = await client.get("/public/endpoints")
    assert response.status_code == 200, response.text
    providers: Final = tuple(ProviderCreateInfo.model_validate(entry) for entry in response.json())
    matches: Final = tuple(provider for provider in providers if provider.litellm_provider == "tsubasa")
    assert len(matches) == 1
    provider: Final = matches[0]
    assert provider.provider == provider.provider_display_name == "Tsubasa"
    assert provider.default_model_placeholder is not None
    assert get_model_info(provider.default_model_placeholder)["litellm_provider"] == provider.litellm_provider
    registration: Final = JSONProviderRegistry.get(provider.litellm_provider)
    assert registration is not None
    assert {
        field.key: (field.field_type, field.required, field.default_value) for field in provider.credential_fields
    } == {
        "api_key": ("password", True, None),
        "api_base": ("text", False, registration.base_url),
    }
    assert endpoints_response.status_code == 200, endpoints_response.text
    endpoints: Final = SupportedEndpointsResponse.model_validate(endpoints_response.json()).endpoints
    assert [
        endpoint.key for endpoint in endpoints if any(p.slug == provider.litellm_provider for p in endpoint.providers)
    ] == ["chat_completions"]
