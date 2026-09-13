import asyncio
import re
from collections.abc import Callable
from functools import cache
from typing import TYPE_CHECKING, Any, Final
from urllib.parse import urlparse

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    get_last_user_message,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.secret_managers.get_azure_ad_token_provider import (
    get_azure_ad_token_provider,
)

if TYPE_CHECKING:
    from litellm.types.llms.openai import AllMessageValues

# Azure Content Safety APIs have a 10,000 character limit per request.
AZURE_CONTENT_SAFETY_MAX_TEXT_LENGTH: Final = 10000

# Azure Content Safety bills text in 1,000-character "text records"; a submitted
# chunk of N characters consumes ceil(N / 1000) text records.
AZURE_CONTENT_SAFETY_TEXT_RECORD_LENGTH: Final = 1000

AZURE_CONTENT_SAFETY_ENTRA_SCOPE: Final = "https://cognitiveservices.azure.com/.default"

AZURE_CONTENT_SAFETY_ENTRA_HOST_SUFFIXES: Final = (
    ".cognitiveservices.azure.com",
    ".cognitiveservices.azure.us",
    ".cognitiveservices.azure.cn",
    ".services.ai.azure.com",
)


def _assert_entra_destination_is_azure(api_base: str) -> None:
    """An Entra token is scoped to every Cognitive Services resource the identity can reach,
    not to one resource, so it is only ever sent to an Azure endpoint over TLS. Entra also
    requires the resource's custom subdomain, so any other host is not a valid destination."""
    parsed: Final = urlparse(api_base)
    host: Final = (parsed.hostname or "").lower()
    if parsed.scheme == "https" and host.endswith(AZURE_CONTENT_SAFETY_ENTRA_HOST_SUFFIXES):
        return
    raise ValueError(
        f"Azure Content Safety: refusing to send a Microsoft Entra token to api_base {api_base!r}. "
        "Entra authentication needs the resource's HTTPS custom subdomain endpoint, for example "
        "https://your-resource.cognitiveservices.azure.com. Set api_key to reach any other host"
    )


@cache
def _default_entra_token_provider() -> Callable[[], str]:
    """Entra token provider for the Content Safety data plane, one credential per process."""
    try:
        return get_azure_ad_token_provider(azure_scope=AZURE_CONTENT_SAFETY_ENTRA_SCOPE)
    except ImportError as e:
        raise ValueError(
            "Azure Content Safety: api_key is not set and azure-identity is not installed. "
            "Set api_key, or install azure-identity to authenticate with Microsoft Entra ID"
        ) from e


class AzureGuardrailBase:
    """
    Base class for Azure guardrails.

    Provides shared initialisation (API credentials, HTTP client) and
    utilities (text splitting, authenticated POST) used by all Azure
    Content Safety guardrails.
    """

    def __init__(
        self,
        *,
        api_base: str,
        api_key: str | None = None,
        entra_token_provider: Callable[[], str] | None = None,
        **kwargs: Any,
    ):
        # Forward remaining kwargs to the next class in the MRO
        # (typically CustomGuardrail).
        super().__init__(**kwargs)

        self.async_handler = get_async_httpx_client(llm_provider=httpxSpecialProvider.GuardrailCallback)
        self.api_key = api_key
        self.api_base = api_base
        self.api_version: str = kwargs.get("api_version") or "2024-09-01"
        if not api_key:
            _assert_entra_destination_is_azure(api_base)
        self._entra_token_provider: Final = entra_token_provider or (
            None if api_key else _default_entra_token_provider()
        )

    async def _auth_header(self) -> tuple[str, str]:
        """Credential header name and value for a single request.

        Azure Content Safety accepts an API key or a Microsoft Entra token and rejects each
        on the other's header, so exactly one is sent.
        """
        if self.api_key:
            return ("Ocp-Apim-Subscription-Key", self.api_key)

        _assert_entra_destination_is_azure(self.api_base)
        minter: Final = self._entra_token_provider or _default_entra_token_provider()
        try:
            token: Final = await asyncio.to_thread(minter)
        except Exception as e:
            verbose_proxy_logger.exception("Azure Content Safety: Entra token request failed")
            raise ValueError(
                "Azure Content Safety: no credential available. Set api_key, or configure an Entra "
                "identity (AZURE_CLIENT_ID / AZURE_CLIENT_SECRET / AZURE_TENANT_ID, workload identity, "
                "managed identity, or az login) holding the Cognitive Services User role on the resource"
            ) from e
        return ("Authorization", f"Bearer {token}")

    async def _post_to_content_safety(self, endpoint_path: str, request_body: dict[str, object]) -> dict[str, Any]:
        """POST to an Azure Content Safety endpoint with standard auth headers.

        Args:
            endpoint_path: The API action, e.g. ``"text:shieldPrompt"`` or
                ``"text:analyze"``.
            request_body: JSON-serialisable request payload.

        Returns:
            Parsed JSON response dict.
        """
        url: Final = f"{self.api_base}/contentsafety/{endpoint_path}?api-version={self.api_version}"
        auth_name, auth_value = await self._auth_header()
        headers: Final = {  # mutable-ok: AsyncHTTPHandler.post types headers as `dict | None`
            auth_name: auth_value,
            "Content-Type": "application/json",
        }

        verbose_proxy_logger.debug("Azure Content Safety request [%s]: %s", endpoint_path, request_body)
        response: Final = await self.async_handler.post(
            url=url,
            headers=headers,
            json=request_body,
        )
        response_json: Final[dict[str, Any]] = response.json()
        verbose_proxy_logger.debug("Azure Content Safety response [%s]: %s", endpoint_path, response_json)
        return response_json

    @staticmethod
    def split_text_by_words(text: str, max_length: int) -> list[str]:
        """
        Split text into chunks at word boundaries without breaking words.

        Always returns at least one chunk.  Short text (≤ max_length) is
        returned as a single-element list so callers can use a uniform
        loop without branching on length.

        Args:
            text: The text to split
            max_length: Maximum character length of each chunk

        Returns:
            List of text chunks, each not exceeding max_length
        """
        if len(text) <= max_length:
            return [text]

        # Tokenize into alternating non-whitespace and whitespace runs so
        # that original newlines, tabs, and multiple spaces are preserved
        # within each chunk.
        tokens: Final = [match.group(0) for match in re.finditer(r"\S+|\s+", text)]

        chunks: Final[list[str]] = []
        current_chunk = ""

        for token in tokens:
            # Would appending this token exceed the limit?
            if len(current_chunk) + len(token) <= max_length:
                current_chunk += token
            else:
                # Flush whatever we have accumulated so far
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""

                # Force-split any single token longer than max_length
                while len(token) > max_length:
                    chunks.append(token[:max_length])
                    token = token[max_length:]

                current_chunk = token

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def get_user_prompt(self, messages: list["AllMessageValues"]) -> str | None:
        """
        Get the last consecutive block of messages from the user.

        Example:
        messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm good, thank you!"},
            {"role": "user", "content": "What is the weather in Tokyo?"},
        ]
        get_user_prompt(messages) -> "What is the weather in Tokyo?"
        """
        return get_last_user_message(messages)
