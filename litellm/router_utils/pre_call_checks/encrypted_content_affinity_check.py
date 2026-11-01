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

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Final, Optional, cast

from pydantic import BaseModel, TypeAdapter

import litellm
from litellm._logging import verbose_router_logger
from litellm.exceptions import AuthenticationError, BadRequestError
from litellm.integrations.custom_logger import CustomLogger, Span
from litellm.litellm_core_utils.get_llm_provider_logic import declared_provider, get_llm_provider
from litellm.litellm_core_utils.litellm_logging import has_sensitive_keyword
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    anthropic_content_lists,
    encrypted_content_of_block,
    strip_encrypted_reasoning_from_messages,
)
from litellm.litellm_core_utils.routing_common_utils import as_excluded_deployment_ids, as_retry_skipped_deployment_ids
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.router_deprecated import RenamedRouterAttribute, renamed_router_param
from litellm.types.llms.openai import AllMessageValues
from litellm.types.router import Deployment, GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager, get_order_filtered_deployments, with_named_credential

if TYPE_CHECKING:
    from litellm.router_api import RoutingContext

_REQUEST_KWARGS_ADAPTER: Final = TypeAdapter(Mapping[str, object])
_ACCOUNT_SCOPE_HEADERS: Final = frozenset({"openai-organization", "openai-project", "cookie"})
_SINGLE_VENDOR_PROVIDERS: Final = frozenset({LlmProviders.OPENAI.value})


@dataclass(frozen=True, slots=True)
class _ProviderRoute:
    model: str
    provider: str
    api_base: str | None


@dataclass(frozen=True, slots=True)
class _EncryptionBoundary:
    api_base: object
    api_key: object
    model: str | None
    account_scope: frozenset[tuple[str, object]]
    explicit_pair: bool

    def is_shared_with(self, other: "_EncryptionBoundary") -> bool:
        if (self.api_base, self.api_key) != (other.api_base, other.api_key):
            return False
        both_explicit: Final = self.explicit_pair and other.explicit_pair
        return both_explicit or (self.model, self.account_scope) == (other.model, other.account_scope)


class EncryptedContentAffinityCheck(CustomLogger):
    """
    Routes follow-up Responses API requests to the deployment that produced
    the encrypted output items they reference.

    The ``model_id`` is decoded directly from the litellm-encoded item IDs –
    no caching or TTL management needed.

    Wired via ``Router(optional_pre_call_checks=["encrypted_content_affinity"])`` or
    per-model group ``model_group_affinity_config``.
    """

    router: Final = RenamedRouterAttribute()  # deprecated: the router attribute before ``ctx``

    @renamed_router_param("router")
    def __init__(
        self,
        ctx: Optional["RoutingContext"] = None,
        enable_global_affinity: bool = True,
        model_group_affinity_config: dict[str, list[str]] | None = None,
    ) -> None:
        super().__init__()
        self.ctx = ctx
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

        return next(
            (
                model_id
                for item in request_input
                if (model_id := EncryptedContentAffinityCheck._model_id_of_input_item(item)) is not None
            ),
            None,
        )

    @staticmethod
    def _model_id_of_input_item(item: object) -> str | None:
        if not isinstance(item, dict):
            return None

        item_id: Final = item.get("id")
        if item_id and isinstance(item_id, str):
            decoded: Final = ResponsesAPIRequestUtils.decode_encrypted_item_id(item_id)
            if decoded:
                return decoded.get("model_id")

        encrypted_content: Final = item.get("encrypted_content")
        if encrypted_content and isinstance(encrypted_content, str):
            model_id: Final = EncryptedContentAffinityCheck._model_id_from_wrapped_encrypted_content(encrypted_content)
            if model_id:
                return model_id

        return None

    @staticmethod
    def _anthropic_content_blocks(messages: object) -> Iterator[Mapping[str, object]]:
        if not isinstance(messages, list):
            return iter(())
        return (
            cast(Mapping[str, object], block)  # cast-ok: narrowed by isinstance
            for content in anthropic_content_lists(cast(list[object], messages))  # cast-ok: narrowed by isinstance
            for block in cast(list[object], content)  # cast-ok: narrowed by isinstance
            if isinstance(block, Mapping)
        )

    @staticmethod
    def _model_id_from_wrapped_encrypted_content(encrypted_content: str) -> str | None:
        model_id, _ = ResponsesAPIRequestUtils.unwrap_encrypted_content_with_model_id(encrypted_content)
        return model_id or None

    @staticmethod
    def _model_id_of_anthropic_block(block: Mapping[str, object]) -> str | None:
        encrypted_content: Final = encrypted_content_of_block(block)
        if encrypted_content is None:
            return None
        return EncryptedContentAffinityCheck._model_id_from_wrapped_encrypted_content(encrypted_content)

    @staticmethod
    def _extract_model_id_from_anthropic_messages(messages: object) -> str | None:
        return next(
            (
                model_id
                for block in EncryptedContentAffinityCheck._anthropic_content_blocks(messages)
                if (model_id := EncryptedContentAffinityCheck._model_id_of_anthropic_block(block)) is not None
            ),
            None,
        )

    @staticmethod
    def _deployment_model_id(deployment: Mapping[str, object]) -> str | None:
        model_info: Final = deployment.get("model_info")
        if not isinstance(model_info, Mapping):
            return None
        deployment_id: Final = cast(Mapping[str, object], model_info).get("id")  # cast-ok: narrowed by isinstance
        return None if deployment_id is None else str(deployment_id)

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
    def _encryption_boundary_key(
        litellm_params: object,
    ) -> _EncryptionBoundary | None:
        """
        The upstream encryption boundary a deployment's encrypted reasoning belongs to. The values
        are resolved the way the request resolves them, the deployment's own values winning over its
        named credential's, without modifying the deployment. Two explicit ``(api_base, api_key)``
        pairs match on those values alone. On a provider that serves one vendor's models, a deployment
        with no ``api_key`` takes the provider's default key, one with no ``api_base`` takes the endpoint
        its Responses URL builder resolves, and a match involving either also needs the same model and
        the same headers that pick an account.

        Accepts plain dicts (the common case in ``healthy_deployments``) as
        well as ``LiteLLM_Params`` Pydantic instances. A stricter
        ``isinstance(dict)`` guard would silently drop the latter from
        boundary matching and fall back to the full pool — i.e. trigger the
        exact ``invalid_encrypted_content`` failure this check exists to prevent.
        """
        params: Final = EncryptedContentAffinityCheck._params_mapping(litellm_params)
        if params is None:
            return None
        effective_params: Final = with_named_credential(params)
        explicit_api_key: Final = effective_params.get("api_key")
        explicit_api_base: Final = effective_params.get("api_base")
        provider_route: Final = EncryptedContentAffinityCheck._provider_route(effective_params)
        route: Final = (
            provider_route
            if provider_route is not None and provider_route.provider in _SINGLE_VENDOR_PROVIDERS
            else None
        )
        account_scope: Final = EncryptedContentAffinityCheck._account_scope(effective_params)
        if explicit_api_key and explicit_api_base:
            return _EncryptionBoundary(
                explicit_api_base, explicit_api_key, route.model if route else None, account_scope, explicit_pair=True
            )
        if route is None:
            return None
        api_key: Final = explicit_api_key or EncryptedContentAffinityCheck._default_api_key(route)
        if not api_key:
            return None
        api_base: Final = explicit_api_base or EncryptedContentAffinityCheck._default_responses_endpoint(
            route, effective_params
        )
        if not api_base:
            return None
        return _EncryptionBoundary(api_base, api_key, route.model, account_scope, explicit_pair=False)

    @staticmethod
    def _params_mapping(litellm_params: object) -> Mapping[str, object] | None:
        if isinstance(litellm_params, BaseModel):
            return litellm_params.model_dump(exclude_none=True)
        if isinstance(litellm_params, Mapping):
            return _REQUEST_KWARGS_ADAPTER.validate_python(litellm_params)
        return None

    @staticmethod
    def _account_scope(litellm_params: Mapping[str, object]) -> frozenset[tuple[str, object]]:
        return frozenset(
            (name.lower(), str(value))
            for name, value in EncryptedContentAffinityCheck._header_items(litellm_params)
            if name.lower() in _ACCOUNT_SCOPE_HEADERS or has_sensitive_keyword(name)
        )

    @staticmethod
    def _header_items(litellm_params: Mapping[str, object]) -> Iterator[tuple[str, object]]:
        headers: Final = EncryptedContentAffinityCheck._header_mapping(litellm_params.get("headers"))
        extra_headers: Final = EncryptedContentAffinityCheck._header_mapping(litellm_params.get("extra_headers"))
        overridden: Final = frozenset(name.lower() for name in extra_headers)
        yield from ((name, value) for name, value in headers.items() if name.lower() not in overridden)
        yield from extra_headers.items()

    @staticmethod
    def _header_mapping(headers: object) -> Mapping[str, object]:
        if isinstance(headers, Mapping):
            return _REQUEST_KWARGS_ADAPTER.validate_python(headers)
        return {}

    @staticmethod
    def _provider_route(litellm_params: Mapping[str, object]) -> _ProviderRoute | None:
        model: Final = litellm_params.get("model")
        if not isinstance(model, str):
            return None
        custom_llm_provider: Final = litellm_params.get("custom_llm_provider")
        declared: Final = declared_provider(
            model, custom_llm_provider if isinstance(custom_llm_provider, str) else None
        )
        if declared in litellm.provider_list and declared not in _SINGLE_VENDOR_PROVIDERS:
            return None
        try:
            provider_model, provider, _, provider_api_base = get_llm_provider(
                model=model, litellm_params=GenericLiteLLMParams.model_validate(dict(litellm_params))
            )
        except BadRequestError:
            return None
        return _ProviderRoute(provider_model, provider, provider_api_base)

    @staticmethod
    def _default_api_key(route: _ProviderRoute) -> str | None:
        model_info: Final = ProviderConfigManager.get_provider_model_info(
            model=route.model, provider=LlmProviders(route.provider)
        )
        if model_info is None:
            return None
        try:
            return model_info.get_api_key()
        except NotImplementedError:
            return None

    @staticmethod
    def _default_responses_endpoint(route: _ProviderRoute, litellm_params: Mapping[str, object]) -> str | None:
        responses_config: Final = ProviderConfigManager.get_provider_responses_api_config(
            provider=route.provider, model=route.model
        )
        if responses_config is None:
            return None
        try:
            return responses_config.get_complete_url(  # pyright: ignore[reportUnknownMemberType]  # the base types litellm_params as a bare dict
                api_base=route.api_base, litellm_params=dict(litellm_params)
            )
        except (ValueError, BaseLLMException, AuthenticationError):
            return None

    def _find_deployments_on_same_encryption_boundary(
        self,
        healthy_deployments: list[dict],
        model_id: str,
    ) -> tuple[list[dict], Deployment | None]:
        """
        Deployments in ``healthy_deployments`` sharing the originating
        deployment's encryption boundary, alongside the originating
        deployment object (or ``None`` if it was removed / router unavailable).
        """
        if self.ctx is None:
            return [], None
        originating: Final = self.ctx.registry().deployment(model_id)
        if originating is None:
            return [], None
        boundary: Final = self._encryption_boundary_key(originating.litellm_params.model_dump(exclude_none=True))
        if boundary is None:
            return [], originating
        matches: Final = [
            d
            for d in healthy_deployments
            if (candidate := self._encryption_boundary_key(d.get("litellm_params", {}))) is not None
            and boundary.is_shared_with(candidate)
        ]
        return matches, originating

    @staticmethod
    @renamed_router_param("router")
    def strip_reasoning_the_targets_cannot_decrypt(
        ctx: "RoutingContext | None",
        request_input: object,
        anthropic_messages: object,
        target_deployments: Sequence[Mapping[str, object]],
        *,
        unmarked_origin: str | None,
    ) -> None:
        """
        Drop the encrypted reasoning that none of ``target_deployments`` minted or shares an
        encryption boundary with, keeping each item's readable summary. Encrypted reasoning that
        carries no litellm origin marker is attributed to ``unmarked_origin``: the affinity pin
        names the deployment its marker decoded to, a fallback hop names the deployment that just
        failed, and ``None`` drops it, since no deployment is known to have minted it.
        """
        target_ids: Final = frozenset(
            str(model_info["id"])
            for target in target_deployments
            if isinstance((model_info := target.get("model_info")), Mapping) and model_info.get("id") is not None
        )

        @cache
        def target_boundaries() -> tuple[_EncryptionBoundary, ...]:
            return tuple(
                boundary
                for target in target_deployments
                if (boundary := EncryptedContentAffinityCheck._encryption_boundary_key(target.get("litellm_params")))
                is not None
            )

        @cache
        def target_can_decrypt(marked_origin: str | None) -> bool:
            origin_model_id: Final = marked_origin if marked_origin is not None else unmarked_origin
            if origin_model_id is None:
                return False
            if origin_model_id in target_ids:
                return True
            if ctx is None:
                return False
            origin: Final = ctx.registry().deployment(origin_model_id)
            origin_boundary: Final = (
                EncryptedContentAffinityCheck._encryption_boundary_key(
                    origin.litellm_params.model_dump(exclude_none=True)
                )
                if origin is not None
                else None
            )
            return origin_boundary is not None and any(
                origin_boundary.is_shared_with(target_boundary) for target_boundary in target_boundaries()
            )

        def should_strip_input_item(item: Mapping[str, object]) -> bool:
            return not target_can_decrypt(EncryptedContentAffinityCheck._model_id_of_input_item(item))

        def should_strip_anthropic_block(block: Mapping[str, object]) -> bool:
            return not target_can_decrypt(EncryptedContentAffinityCheck._model_id_of_anthropic_block(block))

        ResponsesAPIRequestUtils.strip_encrypted_reasoning_from_input(
            request_input, should_strip=should_strip_input_item
        )
        strip_encrypted_reasoning_from_messages(anthropic_messages, should_strip=should_strip_anthropic_block)

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
        ``model_id`` and pin the request to that deployment. When the origin cannot
        serve this turn and no encryption-boundary peer is configured (it is
        unhealthy, the request was routed to a different group by an auto-router tier
        change or model switch, or the marker is removed/unknown/forged), the
        encrypted reasoning is stripped and the request dispatches to the healthy
        pool with its readable history instead of failing. An order-based fallback hop
        carries ``_target_order``, and the pin only considers deployments of that order,
        so the hop reaches the next order with the origin's reasoning stripped instead
        of replaying it to a deployment that cannot decrypt it.
        """
        request_kwargs = request_kwargs or {}
        typed_healthy_deployments: Final = cast(list[dict[str, object]], healthy_deployments)
        if not self._is_enabled_for_model_group(model):
            return typed_healthy_deployments
        typed_request_kwargs: Final = _REQUEST_KWARGS_ADAPTER.validate_python(request_kwargs)
        target_order: Final = typed_request_kwargs.get("_target_order")
        candidates: Final = (
            get_order_filtered_deployments(typed_healthy_deployments, target_order=target_order)
            if isinstance(target_order, int)
            else typed_healthy_deployments
        )

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

        previous_response_id: Final = typed_request_kwargs.get("previous_response_id")
        previous_response_origin: Final = ResponsesAPIRequestUtils.get_model_id_from_response_id(
            None if previous_response_id is None else str(previous_response_id)
        )
        previous_response_binds_origin: Final = previous_response_origin == model_id
        excluded_ids: Final[frozenset[str]] = (
            frozenset()
            if previous_response_binds_origin
            else frozenset(as_excluded_deployment_ids(typed_request_kwargs.get("_excluded_deployment_ids")))
        )
        retry_skipped_ids: Final[frozenset[str]] = (
            frozenset()
            if previous_response_binds_origin
            else frozenset(as_retry_skipped_deployment_ids(typed_request_kwargs.get("_retry_skipped_deployment_ids")))
        )
        not_excluded_candidates: Final = [
            candidate for candidate in candidates if self._deployment_model_id(candidate) not in excluded_ids
        ]
        untried_candidates: Final = [
            candidate
            for candidate in not_excluded_candidates
            if self._deployment_model_id(candidate) not in retry_skipped_ids
        ] or not_excluded_candidates
        deployment: Final = self._find_deployment_by_model_id(
            healthy_deployments=untried_candidates,
            model_id=model_id,
        )
        if deployment is not None:
            verbose_router_logger.debug(
                "EncryptedContentAffinityCheck: pinning -> deployment=%s",
                model_id,
            )
            request_kwargs["_encrypted_content_affinity_pinned"] = True
            self.strip_reasoning_the_targets_cannot_decrypt(
                self.ctx, request_input, anthropic_messages, (deployment,), unmarked_origin=model_id
            )
            return [deployment]

        # Follow-up switched model_name (LIT-2531): pin by Azure resource instead.
        boundary_matches, _originating = self._find_deployments_on_same_encryption_boundary(
            healthy_deployments=untried_candidates,
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
            self.strip_reasoning_the_targets_cannot_decrypt(
                self.ctx, request_input, anthropic_messages, boundary_matches, unmarked_origin=model_id
            )
            return boundary_matches

        # The origin cannot serve this turn and no peer shares its encryption boundary, so its
        # encrypted reasoning can never decrypt here. Strip it, keep the readable history, and dispatch
        # to the healthy pool instead of failing the request. This also denies an authenticated caller a
        # deployment-id existence oracle: a same-group id, a cross-group id, a removed id and a forged
        # marker all strip and dispatch rather than returning distinguishable responses.
        verbose_router_logger.warning(
            "EncryptedContentAffinityCheck: model_id=%s cannot serve group %s and no deployment on the same "
            "encryption boundary is configured; forwarding without its encrypted reasoning",
            model_id[:64],
            model,
        )
        ResponsesAPIRequestUtils.strip_encrypted_reasoning_from_input(request_input)
        strip_encrypted_reasoning_from_messages(anthropic_messages)
        return candidates
