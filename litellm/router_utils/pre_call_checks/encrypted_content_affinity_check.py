"""
Encrypted-content-aware deployment affinity for the Router.

When Codex or other models use `store: false` with `include: ["reasoning.encrypted_content"]`,
the response output items contain encrypted reasoning tokens tied to the originating
organization's API key. If a follow-up request containing those items is routed to a
different deployment (different org), OpenAI rejects it with an `invalid_encrypted_content`
error because the organization_id doesn't match.

This callback solves the problem by encoding the originating deployment's ``model_id``
into the response output items that carry ``encrypted_content``. Two encoding strategies:

1. **Items with IDs**: Encode model_id into the item ID itself (e.g., ``encitem_...``)
2. **Items without IDs** (Codex): Wrap the encrypted_content with model_id metadata
   (e.g., ``litellm_enc:{base64_metadata};{original_encrypted_content}``)

The encoded model_id is decoded on the next request so the router can pin to the correct
deployment without any cache lookup.

Response post-processing (encoding) is handled by
``ResponsesAPIRequestUtils._update_encrypted_content_item_ids_in_response`` which is
called inside ``_update_responses_api_response_id_with_model_id`` in ``responses/utils.py``.

Request pre-processing (ID/content restoration before forwarding to upstream) is handled by
``ResponsesAPIRequestUtils._restore_encrypted_content_item_ids_in_input`` which is called
in ``get_optional_params_responses_api``.

This pre-call check is responsible only for the routing decision: it reads the encoded
``model_id`` from either item IDs or wrapped encrypted_content and pins the request to
the matching deployment.

Safe to enable globally:
- Only activates when encoded markers appear in the request ``input``.
- No effect on embedding models, chat completions, or first-time requests.
- No quota reduction -- first requests are fully load balanced.
- No cache required.
"""

import time
from typing import TYPE_CHECKING, Any, List, Optional, cast

import httpx

from litellm._logging import verbose_router_logger
from litellm.exceptions import (
    BadRequestError,
    RateLimitError,
    ServiceUnavailableError,
)
from litellm.integrations.custom_logger import CustomLogger, Span
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.router_utils.cooldown_cache import CooldownCacheValue
from litellm.types.llms.openai import AllMessageValues

if TYPE_CHECKING:
    from litellm.router import Router


class EncryptedContentAffinityCheck(CustomLogger):
    """
    Routes follow-up Responses API requests to the deployment that produced
    the encrypted output items they reference.

    The ``model_id`` is decoded directly from the litellm-encoded item IDs –
    no caching or TTL management needed.

    Wired via ``Router(optional_pre_call_checks=["encrypted_content_affinity"])``.
    """

    def __init__(self, router: Optional["Router"] = None) -> None:
        super().__init__()
        self.router = router

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_model_id_from_input(request_input: Any) -> Optional[str]:
        """
        Scan ``input`` items for litellm-encoded encrypted-content markers and
        return the ``model_id`` embedded in the first one found.

        Checks both:
        1. Encoded item IDs (encitem_...) - for clients that send IDs
        2. Wrapped encrypted_content (litellm_enc:...) - for clients like Codex that don't send IDs

        ``input`` can be:
        - a plain string  -> no encoded markers
        - a list of items -> check each item's ``id`` and ``encrypted_content`` fields
        """
        if not isinstance(request_input, list):
            return None

        for item in request_input:
            if not isinstance(item, dict):
                continue

            # First, try to decode from item ID (if present)
            item_id = item.get("id")
            if item_id and isinstance(item_id, str):
                decoded = ResponsesAPIRequestUtils._decode_encrypted_item_id(item_id)
                if decoded:
                    return decoded.get("model_id")

            # If no encoded ID, check if encrypted_content itself is wrapped
            encrypted_content = item.get("encrypted_content")
            if encrypted_content and isinstance(encrypted_content, str):
                (
                    model_id,
                    _,
                ) = ResponsesAPIRequestUtils._unwrap_encrypted_content_with_model_id(
                    encrypted_content
                )
                if model_id:
                    return model_id

        return None

    @staticmethod
    def _find_deployment_by_model_id(
        healthy_deployments: List[dict], model_id: str
    ) -> Optional[dict]:
        for deployment in healthy_deployments:
            model_info = deployment.get("model_info")
            if not isinstance(model_info, dict):
                continue
            deployment_model_id = model_info.get("id")
            if deployment_model_id is not None and str(deployment_model_id) == str(
                model_id
            ):
                return deployment
        return None

    @staticmethod
    def _encryption_boundary_key(
        litellm_params: Any,
    ) -> Optional[tuple]:
        """
        ``(api_base, api_key)`` pair identifying an Azure resource. Two
        deployments sharing both are interchangeable for ``encrypted_content``
        follow-ups; Azure rejects content produced by any other resource.

        Accepts any object exposing dict-style ``.get(key, default)``: plain
        dicts (the common case in ``healthy_deployments``) as well as
        ``LiteLLM_Params``-style Pydantic instances, which define a custom
        ``.get()``. A stricter ``isinstance(dict)`` guard would silently drop
        the latter from boundary matching and fall back to the full pool —
        i.e. trigger the exact ``invalid_encrypted_content`` failure this
        check exists to prevent.
        """
        getter = getattr(litellm_params, "get", None)
        if not callable(getter):
            return None
        api_base = getter("api_base")
        api_key = getter("api_key")
        if not api_base or not api_key:
            return None
        return (api_base, api_key)

    def _find_deployments_on_same_encryption_boundary(
        self,
        healthy_deployments: List[dict],
        model_id: str,
    ) -> tuple[List[dict], Any]:
        """
        Deployments in ``healthy_deployments`` sharing the originating
        deployment's ``(api_base, api_key)``, alongside the originating
        deployment object (or ``None`` if it was removed / router unavailable).
        Returns ``([], originating_or_None)`` when no boundary match exists,
        so the caller can reuse the looked-up ``originating`` rather than
        re-querying the router.
        """
        if self.router is None:
            return [], None
        originating = self.router.get_deployment(model_id=model_id)
        if originating is None:
            return [], None
        boundary = self._encryption_boundary_key(
            originating.litellm_params.model_dump(exclude_none=True)
        )
        if boundary is None:
            return [], originating
        matches = [
            d
            for d in healthy_deployments
            if self._encryption_boundary_key(d.get("litellm_params", {})) == boundary
        ]
        return matches, originating

    # ------------------------------------------------------------------
    # Request routing  (pre-call filter)
    # ------------------------------------------------------------------

    async def async_filter_deployments(
        self,
        model: str,
        healthy_deployments: List,
        messages: Optional[List[AllMessageValues]],
        request_kwargs: Optional[dict] = None,
        parent_otel_span: Optional[Span] = None,
    ) -> List[dict]:
        """
        If the request ``input`` contains litellm-encoded item IDs, decode the
        embedded ``model_id`` and pin the request to that deployment. Raises
        ``RateLimitError`` / ``ServiceUnavailableError`` / ``BadRequestError``
        when the originating deployment is unavailable and no encryption-boundary
        peer exists, rather than dispatching a doomed request to a non-peer
        deployment. The 429/503 split mirrors the originating cooldown's status:
        a 429-induced cooldown surfaces as 429 (with ``Retry-After`` set to the
        remaining cooldown window) so OpenAI-compatible clients back off and
        retry after the deployment is eligible again.
        """
        request_kwargs = request_kwargs or {}
        typed_healthy_deployments = cast(List[dict], healthy_deployments)

        # Signal to the response post-processor that encrypted item IDs should be
        # encoded in the output of this request.  Only set the flag when
        # litellm_metadata already exists (Responses API path).  Using
        # setdefault would create an empty litellm_metadata dict for chat
        # completions / embeddings, which breaks tag-based routing because
        # _get_metadata_variable_name_from_kwargs would pick "litellm_metadata"
        # over "metadata" where tags are actually stored.
        if "litellm_metadata" in request_kwargs:
            request_kwargs["litellm_metadata"][
                "encrypted_content_affinity_enabled"
            ] = True

        request_input = request_kwargs.get("input")
        model_id = self._extract_model_id_from_input(request_input)
        if not model_id:
            return typed_healthy_deployments

        verbose_router_logger.debug(
            "EncryptedContentAffinityCheck: decoded model_id=%s from input item IDs",
            model_id,
        )

        deployment = self._find_deployment_by_model_id(
            healthy_deployments=typed_healthy_deployments,
            model_id=model_id,
        )
        if deployment is not None:
            verbose_router_logger.debug(
                "EncryptedContentAffinityCheck: pinning -> deployment=%s",
                model_id,
            )
            request_kwargs["_encrypted_content_affinity_pinned"] = True
            return [deployment]

        # Follow-up switched model_name (LIT-2531): pin by Azure resource instead.
        boundary_matches, originating = (
            self._find_deployments_on_same_encryption_boundary(
                healthy_deployments=typed_healthy_deployments,
                model_id=model_id,
            )
        )
        if boundary_matches:
            verbose_router_logger.debug(
                "EncryptedContentAffinityCheck: model_id=%s not in healthy_deployments; "
                "pinning to %d deployment(s) on same encryption boundary",
                model_id,
                len(boundary_matches),
            )
            request_kwargs["_encrypted_content_affinity_pinned"] = True
            return boundary_matches

        # Dispatching to a non-peer would guarantee an upstream
        # `invalid_encrypted_content` 400, so fail fast with a clearer error.
        raise await self._unavailable_origin_error(
            model=model,
            model_id=model_id,
            originating=originating,
            parent_otel_span=parent_otel_span,
        )

    async def _unavailable_origin_error(
        self,
        model: str,
        model_id: str,
        originating: Any,
        parent_otel_span: Optional[Span],
    ) -> Exception:
        # Public error messages intentionally omit the originating ``model_id`` so
        # an authenticated caller forging encrypted-content markers cannot use the
        # error surface to enumerate which deployment IDs exist on this router.
        if originating is None:
            return BadRequestError(
                message=(
                    "The deployment that produced this encrypted_content is no "
                    "longer configured on this router, and no deployment on the "
                    "same encryption boundary is available. Re-issue the request "
                    "without the stale encrypted_content items, or restore the "
                    "originating deployment."
                ),
                model=model,
                llm_provider="",
            )

        cooldown = await self._get_origin_cooldown(
            model_id=model_id, parent_otel_span=parent_otel_span
        )

        if cooldown is not None and str(cooldown.get("status_code")) == "429":
            retry_after = self._cooldown_seconds_remaining(cooldown)
            return RateLimitError(
                message=(
                    "The deployment that produced this encrypted_content is "
                    f"rate-limited (cooling down for ~{retry_after}s), and no "
                    "deployment on the same encryption boundary is configured. "
                    "Retry after the Retry-After window or configure a deployment "
                    "with the same (api_base, api_key)."
                ),
                llm_provider="",
                model=model,
                response=httpx.Response(
                    status_code=429,
                    headers={"retry-after": str(retry_after)},
                    request=httpx.Request("POST", "https://litellm.ai/"),
                ),
            )

        return ServiceUnavailableError(
            message=(
                "The deployment that produced this encrypted_content is "
                "currently unavailable (likely cooled down), and no deployment "
                "on the same encryption boundary is configured. Retry later or "
                "configure a deployment with the same (api_base, api_key)."
            ),
            llm_provider="",
            model=model,
        )

    async def _get_origin_cooldown(
        self,
        model_id: str,
        parent_otel_span: Optional[Span],
    ) -> Optional[CooldownCacheValue]:
        if self.router is None:
            return None
        cooldown_cache = getattr(self.router, "cooldown_cache", None)
        if cooldown_cache is None:
            return None
        try:
            active = await cooldown_cache.async_get_active_cooldowns(
                model_ids=[model_id], parent_otel_span=parent_otel_span
            )
        except Exception:
            return None
        for cached_model_id, value in active:
            if cached_model_id == model_id:
                return value
        return None

    @staticmethod
    def _cooldown_seconds_remaining(cooldown: CooldownCacheValue) -> int:
        remaining = (
            float(cooldown.get("timestamp", 0.0))
            + float(cooldown.get("cooldown_time", 0.0))
            - time.time()
        )
        return max(1, int(remaining))
