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
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Final, Optional, Protocol, cast

import httpx

from litellm._logging import verbose_router_logger
from litellm.exceptions import (
    RateLimitError,
    ServiceUnavailableError,
)
from litellm.integrations.custom_logger import CustomLogger, Span
from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    encrypted_content_of_block,
    strip_encrypted_reasoning_from_messages,
)
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.router_utils.cooldown_cache import CooldownCacheValue
from litellm.types.llms.openai import AllMessageValues
from litellm.types.router import Deployment

if TYPE_CHECKING:
    from litellm.router import Router


class _SupportsActiveCooldowns(Protocol):
    """Cooldown-cache handle: this check only reads back the currently active cooldowns."""

    async def async_get_active_cooldowns(
        self, model_ids: list[str], parent_otel_span: Span | None
    ) -> list[tuple[str, CooldownCacheValue]]: ...


class EncryptedContentAffinityCheck(CustomLogger):
    """
    Routes follow-up Responses API requests to the deployment that produced
    the encrypted output items they reference.

    The ``model_id`` is decoded directly from the litellm-encoded item IDs –
    no caching or TTL management needed.

    Wired via ``Router(optional_pre_call_checks=["encrypted_content_affinity"])`` or
    per-model group ``model_group_affinity_config``.
    """

    def __init__(
        self,
        router: Optional["Router"] = None,
        enable_global_affinity: bool = True,
        model_group_affinity_config: dict[str, list[str]] | None = None,
    ) -> None:
        super().__init__()
        self.router = router
        self.enable_global_affinity = enable_global_affinity
        self.model_group_affinity_config: dict[str, list[str]] = model_group_affinity_config or {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def has_model_group_affinity_enabled(
        model_group_affinity_config: dict[str, list[str]] | None,
    ) -> bool:
        if not model_group_affinity_config:
            return False

        return any("encrypted_content_affinity" in checks for checks in model_group_affinity_config.values())

    def _is_enabled_for_model_group(self, model_group: str) -> bool:
        group_checks: Final = self.model_group_affinity_config.get(model_group)
        return self.enable_global_affinity or (
            group_checks is not None and "encrypted_content_affinity" in group_checks
        )

    @staticmethod
    def _extract_model_id_from_input(request_input: object) -> str | None:
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
                model_id = EncryptedContentAffinityCheck._model_id_from_wrapped_encrypted_content(encrypted_content)
                if model_id:
                    return model_id

        return None

    @staticmethod
    def _anthropic_content_blocks(messages: object) -> Iterator[Mapping[str, object]]:
        if not isinstance(messages, list):
            return iter(())
        return (
            cast(Mapping[str, object], block)  # cast-ok: narrowed by isinstance
            for message in cast(list[object], messages)  # cast-ok: narrowed by isinstance
            if isinstance(message, Mapping)
            for content in (cast(Mapping[str, object], message).get("content"),)  # cast-ok: narrowed by isinstance
            if isinstance(content, list)
            for block in cast(list[object], content)  # cast-ok: narrowed by isinstance
            if isinstance(block, Mapping)
        )

    @staticmethod
    def _model_id_from_wrapped_encrypted_content(encrypted_content: str) -> str | None:
        model_id, _ = ResponsesAPIRequestUtils._unwrap_encrypted_content_with_model_id(encrypted_content)
        return model_id or None

    @staticmethod
    def _extract_model_id_from_anthropic_messages(messages: object) -> str | None:
        return next(
            (
                model_id
                for block in EncryptedContentAffinityCheck._anthropic_content_blocks(messages)
                if (encrypted_content := encrypted_content_of_block(block)) is not None
                if (
                    model_id := EncryptedContentAffinityCheck._model_id_from_wrapped_encrypted_content(
                        encrypted_content
                    )
                )
                is not None
            ),
            None,
        )

    @staticmethod
    def _find_deployment_by_model_id(healthy_deployments: list[dict], model_id: str) -> dict | None:
        for deployment in healthy_deployments:
            model_info = deployment.get("model_info")
            if not isinstance(model_info, dict):
                continue
            deployment_model_id = model_info.get("id")
            if deployment_model_id is not None and str(deployment_model_id) == str(model_id):
                return deployment
        return None

    @staticmethod
    def _request_team_id(request_kwargs: Mapping[str, object]) -> str | None:
        containers: Final = (request_kwargs.get("metadata"), request_kwargs.get("litellm_metadata"))
        team_ids: Final = (c.get("user_api_key_team_id") for c in containers if isinstance(c, Mapping))
        return next((tid for tid in team_ids if isinstance(tid, str)), None)

    def _routed_group_candidate_model_ids(self, request_kwargs: Mapping[str, object], model: str) -> frozenset[str]:
        """
        Deployment ids that could serve this turn's routed ``model``, as the router
        resolves a route (model_group_alias / routing group / model_name / team /
        pattern). Delegates to the router so the full precedence is not re-derived here
        and no deployment ids are written into request kwargs bound for the provider.
        """
        if self.router is None:
            return frozenset()
        return self.router.get_candidate_model_ids_for_route(model=model, team_id=self._request_team_id(request_kwargs))

    @staticmethod
    def _encryption_boundary_key(
        litellm_params: object,
    ) -> tuple[object, object] | None:
        """
        ``(api_base, api_key)`` identifies an upstream encryption boundary.
        The values are resolved from the deployment and its named credential
        without modifying the deployment.

        Accepts any object exposing dict-style ``.get(key, default)``: plain
        dicts (the common case in ``healthy_deployments``) as well as
        ``LiteLLM_Params``-style Pydantic instances, which define a custom
        ``.get()``. A stricter ``isinstance(dict)`` guard would silently drop
        the latter from boundary matching and fall back to the full pool —
        i.e. trigger the exact ``invalid_encrypted_content`` failure this
        check exists to prevent.
        """
        getter: Final = getattr(litellm_params, "get", None)
        if not callable(getter):
            return None
        api_base: Final = getter("api_base")
        api_key: Final = getter("api_key")
        credential_name: Final = getter("litellm_credential_name")
        credential_values: Final[Mapping[str, object] | None] = (
            CredentialAccessor.get_credential_values(credential_name)
            if isinstance(credential_name, str) and credential_name
            else None
        )
        effective_api_base: Final = (
            credential_values.get("api_base")
            if credential_values is not None and "api_base" in credential_values
            else api_base
        )
        effective_api_key: Final = (
            credential_values.get("api_key")
            if credential_values is not None and "api_key" in credential_values
            else api_key
        )
        if not effective_api_base or not effective_api_key:
            return None
        return (effective_api_base, effective_api_key)

    def _find_deployments_on_same_encryption_boundary(
        self,
        healthy_deployments: list[dict],
        model_id: str,
    ) -> tuple[list[dict], Deployment | None]:
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
        originating: Final = self.router.get_deployment(model_id=model_id)
        if originating is None:
            return [], None
        boundary: Final = self._encryption_boundary_key(originating.litellm_params.model_dump(exclude_none=True))
        if boundary is None:
            return [], originating
        matches: Final = [
            d for d in healthy_deployments if self._encryption_boundary_key(d.get("litellm_params", {})) == boundary
        ]
        return matches, originating

    # ------------------------------------------------------------------
    # Request routing  (pre-call filter)
    # ------------------------------------------------------------------

    async def async_filter_deployments(
        self,
        model: str,
        healthy_deployments: list,
        messages: list[AllMessageValues] | None,
        request_kwargs: dict | None = None,
        parent_otel_span: Span | None = None,
    ) -> list[dict]:
        """
        If the request ``input`` contains litellm-encoded item IDs, or its Anthropic
        ``messages`` replay a bridge-tagged thinking block, decode the embedded
        ``model_id`` and pin the request to that deployment. Raises
        ``RateLimitError`` / ``ServiceUnavailableError`` when the originating
        deployment is a member of the routed model group but currently unavailable
        and no encryption-boundary peer exists, rather than dispatching a doomed
        request to a non-peer deployment. When the origin is not a member of the
        routed group (an auto-router tier change, a model switch with no peer, a
        removed deployment, or an unknown/forged marker), the encrypted reasoning is
        stripped and the request dispatches with its readable history instead. The
        429/503 split mirrors the originating cooldown's status:
        a 429-induced cooldown surfaces as 429 (with ``Retry-After`` set to the
        remaining cooldown window) so OpenAI-compatible clients back off and
        retry after the deployment is eligible again.
        """
        request_kwargs = request_kwargs or {}
        typed_healthy_deployments: Final = cast(list[dict], healthy_deployments)
        if not self._is_enabled_for_model_group(model):
            return typed_healthy_deployments

        # Signal to the response post-processor that encrypted item IDs should be
        # encoded in the output of this request.  Only set the flag when
        # litellm_metadata already exists (Responses API path).  Using
        # setdefault would create an empty litellm_metadata dict for chat
        # completions / embeddings, which breaks tag-based routing because
        # _get_metadata_variable_name_from_kwargs would pick "litellm_metadata"
        # over "metadata" where tags are actually stored.
        if "litellm_metadata" in request_kwargs:
            request_kwargs["litellm_metadata"]["encrypted_content_affinity_enabled"] = True

        request_input: Final = request_kwargs.get("input")
        anthropic_messages: Final = messages or request_kwargs.get("messages")
        model_id: Final = self._extract_model_id_from_input(
            request_input
        ) or self._extract_model_id_from_anthropic_messages(anthropic_messages)
        if not model_id:
            return typed_healthy_deployments

        verbose_router_logger.debug(
            "EncryptedContentAffinityCheck: decoded model_id=%s from the request's encrypted content markers",
            model_id,
        )

        deployment: Final = self._find_deployment_by_model_id(
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
        boundary_matches, originating = self._find_deployments_on_same_encryption_boundary(
            healthy_deployments=typed_healthy_deployments,
            model_id=model_id,
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

        # The origin cannot serve this turn's routed group and no peer shares the boundary, so its
        # encrypted reasoning can never decrypt here. Strip it, keep the readable history, and dispatch
        # to the routed group instead of failing. Membership is tested by deployment id against the set
        # the router actually resolved for this route, not by model-group name, so an alias, a
        # provider-qualified spelling, a team-public name, or a pattern route of the same group is not
        # mistaken for a tier change. An unknown origin (a removed deployment, or a forged marker) is
        # treated the same as a cross-group one, which also denies an authenticated caller a
        # deployment-id existence oracle: a real cross-group id and a nonexistent id both strip and
        # dispatch rather than returning distinguishable responses. Only a genuine same-group member
        # that is currently unavailable falls through to the fail-fast, preserving the cooldown contract.
        routed_group_model_ids: Final = (
            self._routed_group_candidate_model_ids(request_kwargs, model) if originating is not None else frozenset()
        )
        if str(model_id) not in routed_group_model_ids:
            verbose_router_logger.debug(
                "EncryptedContentAffinityCheck: model_id=%s is not a candidate for the routed group %s; "
                "forwarding without its encrypted reasoning",
                model_id,
                model,
            )
            ResponsesAPIRequestUtils.strip_encrypted_reasoning_from_input(request_input)
            strip_encrypted_reasoning_from_messages(anthropic_messages)
            return typed_healthy_deployments

        # The origin is a member of the routed group but currently unavailable (cooled down); fail fast
        # rather than dispatching to a non-peer, which would guarantee an upstream 400.
        raise await self._unavailable_origin_error(
            model=model,
            model_id=model_id,
            parent_otel_span=parent_otel_span,
        )

    async def _unavailable_origin_error(
        self,
        model: str,
        model_id: str,
        parent_otel_span: Span | None,
    ) -> Exception:
        # Public error messages intentionally omit the originating ``model_id`` so
        # an authenticated caller forging encrypted-content markers cannot use the
        # error surface to enumerate which deployment IDs exist on this router.
        cooldown: Final = await self._get_origin_cooldown(model_id=model_id, parent_otel_span=parent_otel_span)

        if cooldown is not None and str(cooldown.get("status_code")) == "429":
            retry_after: Final = self._cooldown_seconds_remaining(cooldown)
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
        parent_otel_span: Span | None,
    ) -> CooldownCacheValue | None:
        if self.router is None:
            return None
        cooldown_cache: Final[_SupportsActiveCooldowns | None] = getattr(self.router, "cooldown_cache", None)
        if cooldown_cache is None:
            return None
        try:
            active: Final = await cooldown_cache.async_get_active_cooldowns(
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
        remaining = float(cooldown.get("timestamp", 0.0)) + float(cooldown.get("cooldown_time", 0.0)) - time.time()
        return max(1, int(remaining))
