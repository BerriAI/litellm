import sys
import threading
from typing import Final

import pytest

from litellm.proxy.guardrails.guardrail_hooks.azure import base
from litellm.proxy.guardrails.guardrail_hooks.azure.prompt_shield import (
    AzureContentSafetyPromptShieldGuardrail,
)
from litellm.proxy.guardrails.guardrail_hooks.azure.text_moderation import (
    AzureContentSafetyTextModerationGuardrail,
)

GUARDRAIL_CLASSES: Final = (
    AzureContentSafetyPromptShieldGuardrail,
    AzureContentSafetyTextModerationGuardrail,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("guardrail_class", GUARDRAIL_CLASSES)
async def test_api_key_rides_the_subscription_key_header_and_mints_no_token(
    guardrail_class, api_base, capturing_handler
):
    handler, sent = capturing_handler

    def _must_not_mint() -> str:
        raise AssertionError("Entra token minted despite api_key being set")

    guardrail: Final = guardrail_class(
        guardrail_name="azure-guard",
        api_base=api_base,
        api_key="secret-key",
        entra_token_provider=_must_not_mint,
    )
    guardrail.async_handler = handler

    await guardrail.apply_guardrail(inputs={"texts": ["hello"]}, request_data={}, input_type="request")

    assert len(sent) == 1
    assert sent[0].headers["Ocp-Apim-Subscription-Key"] == "secret-key"
    assert "authorization" not in sent[0].headers
    assert sent[0].headers["Content-Type"] == "application/json"


@pytest.mark.asyncio
@pytest.mark.parametrize("guardrail_class", GUARDRAIL_CLASSES)
async def test_omitted_api_key_rides_an_entra_bearer_token(guardrail_class, api_base, capturing_handler):
    handler, sent = capturing_handler

    guardrail: Final = guardrail_class(
        guardrail_name="azure-guard",
        api_base=api_base,
        entra_token_provider=lambda: "entra-token",
    )
    guardrail.async_handler = handler

    await guardrail.apply_guardrail(inputs={"texts": ["hello"]}, request_data={}, input_type="request")

    assert len(sent) == 1
    assert sent[0].headers["Authorization"] == "Bearer entra-token"
    assert "ocp-apim-subscription-key" not in sent[0].headers
    assert sent[0].headers["Content-Type"] == "application/json"


@pytest.mark.asyncio
async def test_blank_api_key_is_treated_as_absent(api_base, capturing_handler):
    """An `os.environ/` api_key resolving to an empty string must reach Entra, not send a blank key."""
    handler, sent = capturing_handler

    guardrail: Final = AzureContentSafetyPromptShieldGuardrail(
        guardrail_name="azure-guard",
        api_base=api_base,
        api_key="",
        entra_token_provider=lambda: "entra-token",
    )
    guardrail.async_handler = handler

    await guardrail.apply_guardrail(inputs={"texts": ["hello"]}, request_data={}, input_type="request")

    assert sent[0].headers["Authorization"] == "Bearer entra-token"
    assert "ocp-apim-subscription-key" not in sent[0].headers


@pytest.mark.asyncio
async def test_token_minting_failure_reports_the_options_and_sends_nothing(api_base, capturing_handler):
    handler, sent = capturing_handler

    def _fails() -> str:
        raise RuntimeError("no managed identity endpoint found")

    guardrail: Final = AzureContentSafetyPromptShieldGuardrail(
        guardrail_name="azure-guard",
        api_base=api_base,
        entra_token_provider=_fails,
    )
    guardrail.async_handler = handler

    with pytest.raises(ValueError, match="no credential available") as exc_info:
        await guardrail.apply_guardrail(inputs={"texts": ["hello"]}, request_data={}, input_type="request")

    assert sent == []
    assert isinstance(exc_info.value.__cause__, RuntimeError)


@pytest.mark.asyncio
async def test_token_minting_failure_keeps_credential_detail_out_of_the_error(api_base, capturing_handler):
    """azure-identity errors carry tenant and client ids, and this message reaches the API caller."""
    handler, _ = capturing_handler

    def _fails() -> str:
        raise RuntimeError("tenant 11111111-2222-3333-4444-555555555555 rejected the request")

    guardrail: Final = AzureContentSafetyPromptShieldGuardrail(
        guardrail_name="azure-guard",
        api_base=api_base,
        entra_token_provider=_fails,
    )
    guardrail.async_handler = handler

    with pytest.raises(ValueError, match="no credential available") as exc_info:
        await guardrail.apply_guardrail(inputs={"texts": ["hello"]}, request_data={}, input_type="request")

    assert "11111111-2222-3333-4444-555555555555" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_token_is_minted_off_the_event_loop_thread(api_base, capturing_handler):
    """Credential sources block: IMDS probes time out and the az CLI credential spawns a subprocess."""
    handler, _ = capturing_handler
    minting_threads: Final[list[int]] = []  # mutable-ok: callee-filled thread log

    def _record_thread() -> str:
        minting_threads.append(threading.get_ident())
        return "entra-token"

    guardrail: Final = AzureContentSafetyPromptShieldGuardrail(
        guardrail_name="azure-guard",
        api_base=api_base,
        entra_token_provider=_record_thread,
    )
    guardrail.async_handler = handler

    await guardrail.apply_guardrail(inputs={"texts": ["hello"]}, request_data={}, input_type="request")

    assert len(minting_threads) == 1
    assert minting_threads[0] != threading.get_ident()


def test_default_credential_is_built_once_per_process(monkeypatch):
    """Rebuilding it per request costs a fresh credential and token round trip on every scan."""
    builds: Final[list[str]] = []  # mutable-ok: callee-filled scope log

    def _build(azure_scope: str):
        builds.append(azure_scope)
        return lambda: "entra-token"

    monkeypatch.setattr(base, "get_azure_ad_token_provider", _build)

    assert base._default_entra_token_provider() is base._default_entra_token_provider()
    assert builds == [base.AZURE_CONTENT_SAFETY_ENTRA_SCOPE]


@pytest.mark.parametrize("guardrail_class", GUARDRAIL_CLASSES)
def test_keyless_guardrail_names_the_missing_dependency_at_startup(guardrail_class, api_base, monkeypatch):
    monkeypatch.setitem(sys.modules, "azure", None)

    with pytest.raises(ValueError, match="azure-identity"):
        guardrail_class(guardrail_name="azure-guard", api_base=api_base)


@pytest.mark.asyncio
async def test_api_key_guardrail_never_reaches_azure_identity(api_base, capturing_handler, monkeypatch):
    """azure-identity ships in the proxy extra, so a key-based deployment must not depend on it."""
    handler, sent = capturing_handler
    monkeypatch.setitem(sys.modules, "azure", None)

    guardrail: Final = AzureContentSafetyPromptShieldGuardrail(
        guardrail_name="azure-guard",
        api_base=api_base,
        api_key="secret-key",
    )
    guardrail.async_handler = handler

    await guardrail.apply_guardrail(inputs={"texts": ["hello"]}, request_data={}, input_type="request")

    assert sent[0].headers["Ocp-Apim-Subscription-Key"] == "secret-key"


@pytest.mark.parametrize(
    "bad_api_base",
    [
        "http://contoso.cognitiveservices.azure.com",
        "https://contoso.cognitiveservices.azure.com.attacker.example",
        "https://attacker.example",
        "https://australiaeast.api.cognitive.microsoft.com",
    ],
)
def test_keyless_guardrail_refuses_a_non_azure_destination(bad_api_base):
    """The Entra token covers every Cognitive Services resource the identity can reach, so a
    typo or a hijacked host would receive far more than one resource's key would give away."""
    with pytest.raises(ValueError, match="refusing to send a Microsoft Entra token") as exc_info:
        AzureContentSafetyPromptShieldGuardrail(guardrail_name="azure-guard", api_base=bad_api_base)

    assert "api_key" in str(exc_info.value)


@pytest.mark.parametrize(
    "good_api_base",
    [
        "https://contoso.cognitiveservices.azure.com",
        "https://contoso.privatelink.cognitiveservices.azure.com",
        "https://contoso.services.ai.azure.com",
        "https://contoso.cognitiveservices.azure.us",
    ],
)
def test_keyless_guardrail_accepts_azure_content_safety_endpoints(good_api_base):
    guardrail: Final = AzureContentSafetyPromptShieldGuardrail(
        guardrail_name="azure-guard",
        api_base=good_api_base,
        entra_token_provider=lambda: "entra-token",
    )

    assert guardrail.api_base == good_api_base


@pytest.mark.asyncio
async def test_api_key_guardrail_may_use_any_destination(capturing_handler):
    """A key is scoped to one resource, so gateways and test doubles stay reachable with one."""
    handler, sent = capturing_handler

    guardrail: Final = AzureContentSafetyPromptShieldGuardrail(
        guardrail_name="azure-guard",
        api_base="https://gateway.internal.example",
        api_key="secret-key",
    )
    guardrail.async_handler = handler

    await guardrail.apply_guardrail(inputs={"texts": ["hello"]}, request_data={}, input_type="request")

    assert sent[0].headers["Ocp-Apim-Subscription-Key"] == "secret-key"


@pytest.mark.asyncio
async def test_clearing_api_key_cannot_send_a_token_to_a_non_azure_destination(capturing_handler):
    """A guardrail admitted on its api_key must not start minting tokens for that same host."""
    handler, sent = capturing_handler

    guardrail: Final = AzureContentSafetyPromptShieldGuardrail(
        guardrail_name="azure-guard",
        api_base="https://gateway.internal.example",
        api_key="secret-key",
        entra_token_provider=lambda: "entra-token",
    )
    guardrail.async_handler = handler
    guardrail.update_in_memory_litellm_params({"api_key": None})

    with pytest.raises(ValueError, match="refusing to send a Microsoft Entra token"):
        await guardrail.apply_guardrail(inputs={"texts": ["hello"]}, request_data={}, input_type="request")

    assert sent == []
