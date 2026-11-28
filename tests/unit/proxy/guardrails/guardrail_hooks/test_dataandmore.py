import json
import os
from collections.abc import Iterator, Mapping
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import litellm
from litellm.caching.dual_cache import DualCache
from litellm.exceptions import GuardrailRaisedException
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.dataandmore import (
    guardrail_class_registry,
    guardrail_initializer_registry,
    initialize_guardrail,
)
from litellm.proxy.guardrails.guardrail_hooks.dataandmore.dataandmore import (
    DataAndMoreGuardrail,
)
from litellm.proxy.utils import ProxyLogging
from litellm.types.guardrails import SupportedGuardrailIntegrations
from litellm.types.proxy.guardrails.guardrail_hooks.dataandmore import (
    DataAndMoreGuardrailConfigModel,
)
from litellm.types.utils import Choices, GenericGuardrailAPIInputs, Message, ModelResponse

DEFAULT_API_BASE: Final = "https://dgm.dataandmore.com/api/dlp/ai-gateway/litellm"


def test_dataandmore_guard_registry():
    assert SupportedGuardrailIntegrations.DATAANDMORE.value in guardrail_initializer_registry
    assert SupportedGuardrailIntegrations.DATAANDMORE.value in guardrail_class_registry
    assert guardrail_class_registry["dataandmore"].get_config_model().ui_friendly_name() == "Data & More"


def test_dataandmore_config_model_defaults():
    assert DataAndMoreGuardrailConfigModel.ui_friendly_name() == "Data & More"
    assert DataAndMoreGuardrailConfigModel.model_fields["unreachable_fallback"].default == "fail_open"


def test_initialize_guardrail_passes_extra_headers():
    litellm_params = MagicMock()
    litellm_params.api_base = DEFAULT_API_BASE
    litellm_params.api_key = "dgmg_test"
    litellm_params.additional_provider_specific_params = None
    litellm_params.unreachable_fallback = "fail_open"
    litellm_params.fail_on_error = True
    litellm_params.mode = "pre_call"
    litellm_params.default_on = True
    litellm_params.extra_headers = ["x-request-id"]
    litellm_params.timeout = 10.0

    with patch("litellm.logging_callback_manager.add_litellm_callback"):
        instance = initialize_guardrail(
            litellm_params,
            {"guardrail_name": "dataandmore-dlp"},
        )

    assert instance.extra_headers == ["x-request-id"]
    assert instance.timeout == 10.0
    assert instance.unreachable_fallback == "fail_open"


class TestDataAndMoreGuardrail:
    def setup_method(self):
        for key in ["DGM_GUARDRAIL_KEY", "DGM_GUARDRAIL_API_BASE"]:
            if key in os.environ:
                del os.environ[key]

    def teardown_method(self):
        for key in ["DGM_GUARDRAIL_KEY", "DGM_GUARDRAIL_API_BASE"]:
            if key in os.environ:
                del os.environ[key]

    def test_initialization_requires_api_base(self):
        with pytest.raises(ValueError, match="requires api_base"):
            DataAndMoreGuardrail(
                api_key="dgmg_test",
                guardrail_name="dataandmore-dlp",
                event_hook="pre_call",
            )

    def test_initialization_with_env_api_base(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("DGM_GUARDRAIL_API_BASE", DEFAULT_API_BASE)
        monkeypatch.setenv("DGM_GUARDRAIL_KEY", "dgmg_env_key")

        guardrail = DataAndMoreGuardrail(
            guardrail_name="dataandmore-dlp",
            event_hook="pre_call",
            default_on=True,
        )

        assert guardrail.api_base == f"{DEFAULT_API_BASE}/beta/litellm_basic_guardrail_api"
        assert guardrail.headers["x-api-key"] == "dgmg_env_key"
        assert guardrail.unreachable_fallback == "fail_open"

    def test_initialization_with_custom_api_base(self):
        guardrail = DataAndMoreGuardrail(
            api_base="https://eu.example.com/api/dlp/ai-gateway/litellm",
            api_key="dgmg_custom",
            guardrail_name="dataandmore-dlp",
            event_hook="post_call",
            default_on=False,
            unreachable_fallback="fail_open",
            timeout=10.0,
        )

        assert (
            guardrail.api_base
            == "https://eu.example.com/api/dlp/ai-gateway/litellm/beta/litellm_basic_guardrail_api"
        )
        assert guardrail.unreachable_fallback == "fail_open"
        assert guardrail.timeout == 10.0

    def test_get_config_model(self):
        config_model = DataAndMoreGuardrail.get_config_model()
        assert config_model is not None
        assert config_model.__name__ == "DataAndMoreGuardrailConfigModel"
        assert config_model.ui_friendly_name() == "Data & More"

    @pytest.mark.asyncio
    async def test_apply_guardrail_allows_safe_request(self):
        guardrail = DataAndMoreGuardrail(
            api_base=DEFAULT_API_BASE,
            api_key="dgmg_test",
            guardrail_name="dataandmore-dlp",
            event_hook="pre_call",
            default_on=True,
        )

        inputs = GenericGuardrailAPIInputs(texts=["What is the capital of Japan?"])
        request_data = {
            "proxy_server_request": {
                "messages": [{"role": "user", "content": "What is the capital of Japan?"}],
                "model": "gpt-4o-mini",
            }
        }

        mock_response = MagicMock()
        mock_response.json.return_value = {"action": "NONE"}
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            guardrail.async_handler,
            "post",
            new=AsyncMock(return_value=mock_response),
        ) as mock_post:
            result = await guardrail.apply_guardrail(
                inputs=inputs,
                request_data=request_data,
                input_type="request",
                logging_obj=None,
            )

        assert result["texts"] == ["What is the capital of Japan?"]
        mock_post.assert_awaited_once()
        assert mock_post.await_args.kwargs["url"] == f"{DEFAULT_API_BASE}/beta/litellm_basic_guardrail_api"

    @pytest.mark.asyncio
    async def test_apply_guardrail_blocks_on_blocked_action(self):
        guardrail = DataAndMoreGuardrail(
            api_base=DEFAULT_API_BASE,
            api_key="dgmg_test",
            guardrail_name="dataandmore-dlp",
            event_hook="pre_call",
            default_on=True,
        )

        inputs = GenericGuardrailAPIInputs(texts=["My CPR is 010190-1234"])
        request_data = {
            "proxy_server_request": {
                "messages": [{"role": "user", "content": "My CPR is 010190-1234"}],
                "model": "gpt-4o-mini",
            }
        }

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "action": "BLOCKED",
            "blocked_reason": "Sensitive personal data detected",
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            guardrail.async_handler,
            "post",
            new=AsyncMock(return_value=mock_response),
        ):
            with pytest.raises(GuardrailRaisedException, match="Sensitive personal data detected"):
                await guardrail.apply_guardrail(
                    inputs=inputs,
                    request_data=request_data,
                    input_type="request",
                    logging_obj=None,
                )


DENIAL: Final = "synthetic dataandmore denial"


class _DataAndMore:
    def __init__(self, verdict: Mapping[str, str]) -> None:
        self.verdict: Final = verdict
        self.sent: tuple[Mapping[str, object], ...] = ()

    def respond(self, request: httpx.Request) -> httpx.Response:
        self.sent = (*self.sent, json.loads(request.content))
        return httpx.Response(200, json=dict(self.verdict))


@pytest.fixture
def registered(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[_DataAndMore]:
    hook, verdict = request.param
    vendor: Final = _DataAndMore(verdict)
    guardrail: Final = DataAndMoreGuardrail(
        api_base="http://dataandmore.invalid",
        api_key="dgmg_test",
        guardrail_name="dataandmore-dlp",
        event_hook=hook,
        default_on=True,
        unreachable_fallback="fail_open",
    )
    guardrail.async_handler = AsyncHTTPHandler(transport=httpx.MockTransport(vendor.respond))
    monkeypatch.setattr(litellm, "callbacks", [guardrail])
    yield vendor


def _chat_request(text: str) -> dict[str, object]:
    return {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": text}], "metadata": {}}


BLOCKED: Final = {"action": "BLOCKED", "blocked_reason": DENIAL}
ALLOWED: Final = {"action": "NONE"}


@pytest.mark.asyncio
@pytest.mark.parametrize("registered", [("pre_call", BLOCKED)], indirect=True)
async def test_proxy_pre_call_hook_blocks_on_dataandmore_verdict(registered: _DataAndMore) -> None:
    with pytest.raises(GuardrailRaisedException, match=DENIAL):
        await ProxyLogging(user_api_key_cache=DualCache()).pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            data=_chat_request("My CPR is 010190-1234"),
            call_type="completion",
        )

    assert [sent["texts"] for sent in registered.sent] == [["My CPR is 010190-1234"]]


@pytest.mark.asyncio
@pytest.mark.parametrize("registered", [("post_call", BLOCKED)], indirect=True)
async def test_proxy_post_call_hook_blocks_on_dataandmore_verdict(registered: _DataAndMore) -> None:
    response: Final = ModelResponse(choices=[Choices(index=0, message=Message(role="assistant", content="leaked"))])

    with pytest.raises(GuardrailRaisedException, match=DENIAL):
        await ProxyLogging(user_api_key_cache=DualCache()).post_call_success_hook(
            data=_chat_request("hello"), user_api_key_dict=UserAPIKeyAuth(), response=response
        )

    assert [(sent["input_type"], sent["texts"]) for sent in registered.sent] == [("response", ["leaked"])]


@pytest.mark.asyncio
@pytest.mark.parametrize("registered", [("pre_call", ALLOWED)], indirect=True)
async def test_proxy_pre_call_hook_records_one_guardrail_entry_per_check(registered: _DataAndMore) -> None:
    data: Final = _chat_request("hello")

    await ProxyLogging(user_api_key_cache=DualCache()).pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(), data=data, call_type="completion"
    )

    entries: Final = data["litellm_metadata"]["standard_logging_guardrail_information"]
    assert [(entry["guardrail_name"], entry["guardrail_status"]) for entry in entries] == [
        ("dataandmore-dlp", "success")
    ]
    assert len(registered.sent) == 1
