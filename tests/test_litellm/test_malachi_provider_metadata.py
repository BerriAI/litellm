import json
from pathlib import Path

from litellm.types.utils import LlmProviders

PROVIDER_DOCS_PATH = Path("docs/my-website/docs/providers/malachi.md")


def test_llm_providers_includes_malachi():
    assert LlmProviders.MALACHI.value == "malachi"


def test_provider_endpoints_support_includes_malachi():
    payload = json.loads(Path("provider_endpoints_support.json").read_text())

    malachi = payload["providers"]["malachi"]

    assert malachi["display_name"] == "Malachi (`malachi`)"
    assert malachi["url"] == "https://docs.litellm.ai/docs/providers/malachi"
    assert PROVIDER_DOCS_PATH.exists()
    assert malachi["endpoints"] == {
        "chat_completions": True,
        "responses": True,
    }


def test_provider_create_fields_includes_malachi():
    payload = json.loads(
        Path("litellm/proxy/public_endpoints/provider_create_fields.json").read_text()
    )

    malachi = next(item for item in payload if item["litellm_provider"] == "malachi")

    assert malachi["provider"] == "MALACHI"
    assert malachi["provider_display_name"] == "Malachi"
    assert malachi["default_model_placeholder"] == "malachi/my-model"

    credential_fields = {field["key"]: field for field in malachi["credential_fields"]}
    assert set(credential_fields) == {"api_base", "api_key"}
    assert credential_fields["api_base"]["field_type"] == "text"
    assert credential_fields["api_key"]["field_type"] == "password"
    assert (
        credential_fields["api_base"]["placeholder"] in (None, "https://your-malachi-host/v1")
    )
    assert credential_fields["api_base"]["default_value"] is None

    serialized = json.dumps(malachi)
    assert "tailnet.finalbuildgames.com" not in serialized
    assert "finalbuildgames" not in serialized
