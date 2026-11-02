from collections.abc import Mapping
from types import MappingProxyType
from typing import Final
from unittest.mock import MagicMock

import pytest

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.azure.live.handler import AzureOpenAILiveSessions
from litellm.llms.openai.live.handler import LiveClientWebSocket


class _RecordingLiveSessions:
    def __init__(self) -> None:
        self.kwargs: Mapping[str, object] | None = None

    async def async_live_session(
        self,
        *,
        model: str,
        websocket: LiveClientWebSocket,
        logging_obj: Logging,
        session_start: Mapping[str, object],
        api_base: str | None,
        api_key: str | None,
        upstream_url: str | None = None,
        additional_headers: Mapping[str, str] | None = None,
    ) -> None:
        self.kwargs = MappingProxyType(
            {
                "model": model,
                "websocket": websocket,
                "logging_obj": logging_obj,
                "session_start": session_start,
                "api_base": api_base,
                "api_key": api_key,
                "upstream_url": upstream_url,
                "additional_headers": additional_headers,
            }
        )


@pytest.mark.parametrize(
    ("api_base", "expected_url"),
    [
        (
            "https://voice-resource.openai.azure.com",
            "wss://voice-resource.openai.azure.com/openai/v1/live/sessions",
        ),
        (
            "https://voice-resource.services.ai.azure.com/openai/deployments/ignored?api-version=preview",
            "wss://voice-resource.services.ai.azure.com/openai/v1/live/sessions",
        ),
        (
            "http://localhost:8080/azure",
            "ws://localhost:8080/openai/v1/live/sessions",
        ),
    ],
)
def test_construct_live_url_uses_azure_ga_path(api_base: str, expected_url: str) -> None:
    assert AzureOpenAILiveSessions.construct_live_url(api_base) == expected_url


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("api_key", "azure_ad_token", "expected_headers", "expected_credential"),
    [
        ("azure-key", None, {"api-key": "azure-key"}, "azure-key"),
        (None, "entra-token", {"Authorization": "Bearer entra-token"}, "entra-token"),
    ],
)
async def test_async_live_session_uses_azure_endpoint_and_authentication(
    api_key: str | None,
    azure_ad_token: str | None,
    expected_headers: Mapping[str, str],
    expected_credential: str,
) -> None:
    relay: Final = _RecordingLiveSessions()
    handler: Final = AzureOpenAILiveSessions(relay)
    websocket: Final = MagicMock(spec=LiveClientWebSocket)
    logging_obj: Final = MagicMock(spec=Logging)
    session_start: Final[Mapping[str, object]] = MappingProxyType(
        {
            "type": "session.start",
            "session": {"model": "azure-live-alias"},
        }
    )

    await handler.async_live_session(
        model="gpt-live-1-deployment",
        websocket=websocket,
        logging_obj=logging_obj,
        session_start=session_start,
        api_base="https://voice-resource.openai.azure.com",
        api_key=api_key,
        azure_ad_token=azure_ad_token,
    )

    assert relay.kwargs == {
        "model": "gpt-live-1-deployment",
        "websocket": websocket,
        "logging_obj": logging_obj,
        "session_start": session_start,
        "api_base": "https://voice-resource.openai.azure.com",
        "api_key": expected_credential,
        "upstream_url": "wss://voice-resource.openai.azure.com/openai/v1/live/sessions",
        "additional_headers": expected_headers,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("api_base", "api_key", "azure_ad_token", "expected_message"),
    [
        (None, "azure-key", None, "api_base is required for Azure OpenAI Live session calls"),
        (
            "https://voice-resource.openai.azure.com",
            None,
            None,
            "Missing Azure credentials for the realtime endpoint",
        ),
    ],
)
async def test_async_live_session_rejects_missing_configuration(
    api_base: str | None,
    api_key: str | None,
    azure_ad_token: str | None,
    expected_message: str,
) -> None:
    handler: Final = AzureOpenAILiveSessions(_RecordingLiveSessions())

    with pytest.raises(ValueError, match=expected_message):
        await handler.async_live_session(
            model="gpt-live-1-deployment",
            websocket=MagicMock(spec=LiveClientWebSocket),
            logging_obj=MagicMock(spec=Logging),
            session_start=MappingProxyType({"type": "session.start"}),
            api_base=api_base,
            api_key=api_key,
            azure_ad_token=azure_ad_token,
        )
