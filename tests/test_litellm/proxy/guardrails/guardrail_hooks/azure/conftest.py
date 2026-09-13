from collections.abc import Sequence
from typing import Final

import httpx
import pytest

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy.guardrails.guardrail_hooks.azure.base import (
    _default_entra_token_provider,
)

CLEAN_RESPONSE_FOR_BOTH_GUARDRAILS: Final = {
    "userPromptAnalysis": {"attackDetected": False},
    "documentsAnalysis": [],
    "categoriesAnalysis": [],
    "blocklistsMatch": [],
}


@pytest.fixture(autouse=True)
def clear_default_entra_provider_cache():
    """A credential resolved elsewhere in the session would otherwise be reused here, masking
    the dependency and failure paths these tests assert."""
    _default_entra_token_provider.cache_clear()
    yield
    _default_entra_token_provider.cache_clear()


@pytest.fixture
def api_base() -> str:
    return "https://contoso.cognitiveservices.azure.com"


@pytest.fixture
def capturing_handler() -> tuple[AsyncHTTPHandler, Sequence[httpx.Request]]:
    """An HTTP handler answering every Content Safety call, paired with the requests it saw."""
    sent: Final[list[httpx.Request]] = []  # mutable-ok: callee-filled request log, handed back read-only

    def _record(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json=CLEAN_RESPONSE_FOR_BOTH_GUARDRAILS)

    handler: Final = AsyncHTTPHandler()
    handler.client = httpx.AsyncClient(transport=httpx.MockTransport(_record))
    return handler, sent
