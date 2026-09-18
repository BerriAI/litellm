"""
Nadir as the Complexity Router's classifier.

A decision-only integration: the tier comes from a call to Nadir's ``/v1/bucket``
endpoint, and everything else stays here. The tier's model pool, the provider call, the
operator's own provider keys, fallbacks, spend tracking and the response path are
untouched, so Nadir sees the messages it is asked to classify and never the completion.

``/v1/bucket`` runs a trained complexity classifier (a fine-tuned encoder, not an LLM
call) and answers ``simple`` / ``medium`` / ``complex``, which map to the router's
default SIMPLE / MEDIUM / COMPLEX tiers. A router whose tiers are renamed with
``tier_labels``, or defined with ``tier_definitions``, passes its own names as
``tier_map``.

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
              SIMPLE: gpt-4o-mini
              MEDIUM: gpt-4o
              COMPLEX: o1-preview

``NADIR_API_KEY`` attributes the decision to an account and lifts the anonymous rate
limit; without it the endpoint still answers, burst-limited per IP and stored nowhere.
``NADIR_API_BASE`` overrides the host for a self-hosted or on-prem deployment.

Every failure mode is the router's existing one: this returns None to decline, and a
network error, a timeout past ``classifier_plugin_timeout_ms`` or an unknown bucket all
hand the request to ``classifier_fallback``. Nothing here can fail a completion.

Privacy note for operators: the messages are sent to the configured Nadir host, which is
a third party unless that host is your own. That is the same disclosure a remote LLM
classifier carries, and it is the reason the local heuristic scorers exist.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.types.llms.custom_http import httpxSpecialProvider
from litellm.types.router import RoutingContext

DEFAULT_NADIR_API_BASE: Final = "https://api.getnadir.com"

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


class NadirComplexityClassifier:
    """Classifier plugin that asks Nadir's decision API which tier a request belongs to.

    Args:
        api_base: Nadir host. Defaults to ``NADIR_API_BASE``, then to the hosted API.
        api_key: Nadir API key. Defaults to ``NADIR_API_KEY``; anonymous when unset.
        tier_map: Nadir bucket name -> the tier name this router routes on. Defaults to the
            router's built-in tier names.
    """

    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        tier_map: Mapping[str, str] | None = None,
    ) -> None:
        self._api_base: Final = api_base
        self._api_key: Final = api_key
        # A plain dict, not a MappingProxyType: the proxy deepcopies a deployment's
        # litellm_params, this instance travels inside them, and a mappingproxy cannot be
        # deepcopied. Router construction would fail before the first request.
        self._tier_map: Final[dict[str, str]] = dict(  # mutable-ok: read-only after __init__, deepcopy-safe
            DEFAULT_TIER_MAP if tier_map is None else tier_map
        )

    def _headers(self) -> dict[str, str]:
        api_key: Final = self._api_key or os.getenv("NADIR_API_KEY")
        return {"X-API-Key": api_key} if api_key else {}  # mutable-ok: request headers, handed straight to httpx

    async def classify(self, context: RoutingContext) -> str | None:
        """Return the tier Nadir places this request in, or None to decline.

        Declines rather than guesses on anything the endpoint cannot grade: a request with no
        messages, and a bucket name outside ``tier_map`` (which is what a renamed tier set looks
        like before ``tier_map`` is configured). Both leave the decision to ``classifier_fallback``,
        where a local scorer still routes the request.
        """
        messages: Final = context.structured_messages or context.raw_messages
        if not messages:
            return None
        api_base: Final = self._api_base or os.getenv("NADIR_API_BASE") or DEFAULT_NADIR_API_BASE
        client: Final = get_async_httpx_client(llm_provider=httpxSpecialProvider.ComplexityClassifier)
        body: Final = {"messages": list(messages), "source": "litellm"}  # mutable-ok: one request body
        response: Final = await client.post(url=_bucket_url(api_base), json=body, headers=self._headers())
        payload: Final = response.json()
        if not isinstance(payload, Mapping):
            return None
        bucket: Final = payload.get("bucket")
        if not isinstance(bucket, str):
            return None
        return self._tier_map.get(bucket.strip().lower())


nadir_classifier: Final = NadirComplexityClassifier()
"""Module-level instance, so the proxy config can name it by dotted path."""
