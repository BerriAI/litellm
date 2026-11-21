from typing import Final
from urllib.parse import urlsplit, urlunsplit

from litellm.llms.azure.realtime.handler import AzureOpenAIRealtime
from litellm.llms.openai.live.handler import LIVE_WEBSOCKET_SCHEMES, LiveUpstream

_AZURE_LIVE_PATH: Final = "/openai/v1/live/sessions"


def azure_live_upstream(api_base: str, api_key: str | None, azure_ad_token: str | None) -> LiveUpstream:
    parsed_url: Final = urlsplit(api_base)
    if not parsed_url.netloc:
        raise ValueError("api_base is required for Azure Live session calls")
    websocket_scheme: Final = LIVE_WEBSOCKET_SCHEMES.get(parsed_url.scheme, parsed_url.scheme)
    return LiveUpstream(
        url=urlunsplit((websocket_scheme, parsed_url.netloc, _AZURE_LIVE_PATH, "", "")),
        headers=AzureOpenAIRealtime.get_auth_headers(api_key=api_key, azure_ad_token=azure_ad_token),
    )
