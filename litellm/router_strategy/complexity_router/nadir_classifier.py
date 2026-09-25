"""
Nadir as the Complexity Router's classifier.

A decision-only integration: the tier comes from a call to Nadir's ``/v1/bucket``
endpoint, and everything else stays here. The tier's model pool, the provider call, the
operator's own provider keys, fallbacks, spend tracking and the response path are
untouched, so Nadir sees the messages it is asked to classify and never the response.

``/v1/bucket`` grades the request without generating an answer and replies ``simple`` /
``medium`` / ``complex``, which map to the router's default SIMPLE / MEDIUM / COMPLEX
tiers. Renaming the tiers with ``tier_labels`` needs nothing here, since the router still
accepts the default names. A router built on ``tier_definitions`` names its own tiers, so
it constructs its own instance with a ``tier_map`` onto them.

Configure it in the proxy by pointing ``classifier_plugin`` at the module-level
instance::

    model_list:
      - model_name: smart-router
        litellm_params:
          model: auto_router/complexity_router
          complexity_router_config:
            classifier_type: custom
            classifier_plugin: litellm.router_strategy.complexity_router.nadir_classifier.nadir_classifier
            tiers:
              SIMPLE: gpt-5-nano
              MEDIUM: gpt-5-mini
              COMPLEX: gpt-5

``NADIR_API_KEY`` attributes decisions to a Nadir account and lifts the anonymous rate
limit. That account's request log then keeps the user text of each classified request, or
only a hash of it when the account has prompt storage turned off. Without a key the
endpoint still answers and keeps no prompt, but its per-IP rate limit is sized for trying
the plugin out, and requests over it route on ``classifier_fallback``.

``NADIR_API_BASE`` points at a self-hosted or on-prem Nadir. ``NADIR_API_KEY`` is only
ever sent to that base, or to the hosted API when it is unset: an ``api_base`` passed in
code brings its own ``api_key``.

Every failure mode is the router's existing one: this returns None to decline, and a
network error, an error status, a timeout past ``classifier_plugin_timeout_ms`` or an
unknown bucket all hand the request to ``classifier_fallback``. Nothing here can fail a
completion.

Privacy note for operators: the messages are sent to the configured Nadir host, which is
a third party unless that host is your own. That is the same disclosure a remote LLM
classifier carries, and it is the reason the local heuristic scorers exist.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from litellm.constants import NADIR_DEFAULT_API_BASE
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, get_async_httpx_client
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.custom_http import httpxSpecialProvider
from litellm.types.router import RoutingContext

DEFAULT_TIER_MAP: Final[Mapping[str, str]] = MappingProxyType(
    {"simple": "SIMPLE", "medium": "MEDIUM", "complex": "COMPLEX"}
)

_API_PATH: Final = "/v1/bucket"


def _bucket_url(api_base: str) -> str:
    """Join the endpoint path to a host, tolerating a base that already ends in ``/v1``.

    Nadir's own docs advertise ``https://api.getnadir.com/v1`` as the base URL, because the
    OpenAI-compatible clients that consume it append ``/chat/completions``. An operator who
    copies that value into ``NADIR_API_BASE`` would otherwise send ``/v1/v1/bucket`` and get a
    404 that reads like the endpoint does not exist.
    """
    base: Final = api_base.rstrip("/").removesuffix("/v1")
    return f"{base}{_API_PATH}"


def _configured_bucket_url() -> str:
    """The endpoint the environment names: ``NADIR_API_BASE``, else the hosted API."""
    return _bucket_url(get_secret_str("NADIR_API_BASE") or NADIR_DEFAULT_API_BASE)


class NadirComplexityClassifier:
    """Classifier plugin that asks Nadir's decision API which tier a request belongs to.

    Args:
        api_base: Nadir host. Defaults to ``NADIR_API_BASE``, then to the hosted API.
        api_key: Nadir API key. Defaults to ``NADIR_API_KEY`` only when the request goes to the
            host the environment names, so the environment key never follows an ``api_base``
            set in code; anonymous otherwise.
        tier_map: Nadir bucket name -> the tier name this router routes on. Defaults to the
            router's built-in tier names.
        client: HTTP client the request goes out on. Defaults to litellm's shared async client.
            Leave it unset on an instance a proxy config names, since the proxy deepcopies it.
    """

    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        tier_map: Mapping[str, str] | None = None,
        client: AsyncHTTPHandler | None = None,
    ) -> None:
        self._api_base: Final = api_base
        self._api_key: Final = api_key
        self._client: Final = client
        # A plain dict, not a MappingProxyType: the proxy deepcopies a deployment's
        # litellm_params, this instance travels inside them, and a mappingproxy cannot be
        # deepcopied. Router construction would fail before the first request.
        self._tier_map: Final[dict[str, str]] = dict(  # mutable-ok: read-only after __init__, deepcopy-safe
            DEFAULT_TIER_MAP if tier_map is None else tier_map
        )

    def _headers(self, url: str) -> dict[str, str]:
        env_key: Final = get_secret_str("NADIR_API_KEY") if url == _configured_bucket_url() else None
        api_key: Final = self._api_key or env_key
        return {"X-API-Key": api_key} if api_key else {}  # mutable-ok: request headers, handed straight to httpx

    async def classify(self, context: RoutingContext) -> str | None:
        """Return the tier Nadir places this request in, or None to decline.

        Declines rather than guesses on anything the endpoint cannot grade: a request with no
        messages, and a bucket name outside ``tier_map`` (which is what a ``tier_definitions``
        router looks like before ``tier_map`` is configured). Both leave the decision to
        ``classifier_fallback``, where a local scorer still routes the request.
        """
        messages: Final = context.structured_messages or context.raw_messages
        if not messages:
            return None
        url: Final = _bucket_url(self._api_base) if self._api_base else _configured_bucket_url()
        client: Final = self._client or get_async_httpx_client(llm_provider=httpxSpecialProvider.ComplexityClassifier)
        body: Final = {"messages": list(messages), "source": "litellm"}  # mutable-ok: one request body
        response: Final = await client.post(url=url, json=body, headers=self._headers(url))
        match response.json():
            case {"bucket": str(bucket)}:
                return self._tier_map.get(bucket.strip().lower())
            case _:
                return None


nadir_classifier: Final = NadirComplexityClassifier()
"""Module-level instance, so the proxy config can name it by dotted path."""
