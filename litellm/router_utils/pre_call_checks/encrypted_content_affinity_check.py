"""
Encrypted-content-aware deployment affinity for the Router.

When Codex or other models use `store: false` with `include: ["reasoning.encrypted_content"]`,
the response output items contain encrypted reasoning tokens tied to the originating
organization's API key. If a follow-up request containing those items is routed to a
different deployment (different org), OpenAI rejects it with an `invalid_encrypted_content`
error because the organization_id doesn't match.

This callback solves the problem by:
1. Tracking output item IDs from Responses API responses and mapping them to the
   deployment (model_id) that produced them.
2. On subsequent requests, scanning the `input` field for known item IDs and pinning
   the request to the originating deployment.

Safe to enable globally:
- Only activates when known item IDs appear in the request `input`.
- No effect on embedding models, chat completions, or first-time requests.
- No quota reduction -- first requests are fully load balanced.
"""

from typing import Any, List, Optional, cast

from litellm._logging import verbose_router_logger
from litellm.caching.dual_cache import DualCache
from litellm.integrations.custom_logger import CustomLogger, Span
from litellm.types.llms.openai import AllMessageValues, ResponsesAPIResponse

_DEFAULT_TTL_SECONDS = 86400  # 24 hours


class EncryptedContentAffinityCheck(CustomLogger):
    """
    Routes follow-up Responses API requests to the deployment that produced
    the encrypted output items they reference.

    Wired via ``Router(optional_pre_call_checks=["encrypted_content_affinity"])``.
    """

    CACHE_KEY_PREFIX = "encrypted_content_affinity:v1"

    def __init__(
        self,
        cache: DualCache,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
    ):
        super().__init__()
        self.cache = cache
        self.ttl_seconds = ttl_seconds

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_output_from_response(
        response_obj: Any,
    ) -> Optional[list]:
        """
        Extract the ``output`` list from a Responses API response, handling
        both ``ResponsesAPIResponse`` objects and plain dicts.
        """
        if isinstance(response_obj, ResponsesAPIResponse):
            return response_obj.output
        if isinstance(response_obj, dict) and "output" in response_obj:
            output = response_obj["output"]
            if isinstance(output, list):
                return output
        if hasattr(response_obj, "output"):
            output = response_obj.output
            if isinstance(output, list):
                return output
        return None

    @staticmethod
    def _extract_item_ids_from_output(
        output: list,
    ) -> List[str]:
        """Extract item IDs from output items that contain encrypted_content."""
        item_ids: List[str] = []
        for item in output:
            item_id: Optional[str] = None
            has_encrypted_content = False
            
            if isinstance(item, dict):
                item_id = item.get("id")
                has_encrypted_content = "encrypted_content" in item
            else:
                item_id = getattr(item, "id", None)
                has_encrypted_content = hasattr(item, "encrypted_content")
            
            if item_id and isinstance(item_id, str) and has_encrypted_content:
                item_ids.append(item_id)
        return item_ids

    @staticmethod
    def _extract_item_ids_from_input(request_input: Any) -> List[str]:
        """
        Extract item IDs from input items that contain encrypted_content.

        ``input`` can be:
        - a plain string  -> no item IDs
        - a list of items -> only extract IDs from items with encrypted_content
        """
        if not isinstance(request_input, list):
            return []

        item_ids: List[str] = []
        for item in request_input:
            if isinstance(item, dict):
                item_id = item.get("id")
                has_encrypted_content = "encrypted_content" in item
                if item_id and isinstance(item_id, str) and has_encrypted_content:
                    item_ids.append(item_id)
        return item_ids

    @classmethod
    def _cache_key(cls, item_id: str) -> str:
        return f"{cls.CACHE_KEY_PREFIX}:{item_id}"

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
    def _get_model_id_from_kwargs(kwargs: dict) -> Optional[str]:
        """
        Extract the deployment model_id from success-callback kwargs.

        The Router populates ``litellm_params.metadata.model_info.id`` after
        selecting a deployment. Also check top-level ``model_info`` as a
        fallback (some call paths set it there).
        """
        # Primary path: litellm_params -> metadata -> model_info -> id
        litellm_params = kwargs.get("litellm_params")
        if isinstance(litellm_params, dict):
            metadata = litellm_params.get("metadata")
            if isinstance(metadata, dict):
                model_info = metadata.get("model_info")
                if isinstance(model_info, dict):
                    model_id = model_info.get("id")
                    if model_id is not None:
                        return str(model_id)

        # Fallback: top-level model_info (set by some router call paths)
        model_info = kwargs.get("model_info")
        if isinstance(model_info, dict):
            model_id = model_info.get("id")
            if model_id is not None:
                return str(model_id)

        return None

    # ------------------------------------------------------------------
    # Response tracking  (success callback)
    # ------------------------------------------------------------------

    async def async_log_success_event(
        self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        """
        After a successful Responses API call, cache each output item ID
        mapped to the deployment that produced it.
        """
        output = self._get_output_from_response(response_obj)
        if output is None:
            return

        model_id = self._get_model_id_from_kwargs(kwargs)
        if not model_id:
            verbose_router_logger.debug(
                "EncryptedContentAffinityCheck: model_id not found in kwargs, skipping tracking",
            )
            return

        item_ids = self._extract_item_ids_from_output(output)
        if not item_ids:
            return

        for item_id in item_ids:
            try:
                cache_key = self._cache_key(item_id)
                await self.cache.async_set_cache(
                    cache_key,
                    model_id,
                    ttl=self.ttl_seconds,
                )
            except Exception as e:
                verbose_router_logger.error(
                    "EncryptedContentAffinityCheck: failed to cache item_id=%s error=%s",
                    item_id,
                    e,
                )

        verbose_router_logger.debug(
            "EncryptedContentAffinityCheck: cached %d item IDs -> deployment=%s",
            len(item_ids),
            model_id,
        )

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
        If the request ``input`` contains items whose IDs were previously
        tracked, pin the request to the deployment that produced them.
        """
        request_kwargs = request_kwargs or {}
        typed_healthy_deployments = cast(List[dict], healthy_deployments)

        request_input = request_kwargs.get("input")
        input_item_ids = self._extract_item_ids_from_input(request_input)
        if not input_item_ids:
            return typed_healthy_deployments

        verbose_router_logger.debug(
            "EncryptedContentAffinityCheck: found %d item IDs in input, checking cache",
            len(input_item_ids),
        )

        for item_id in input_item_ids:
            cache_key = self._cache_key(item_id)
            try:
                cached_model_id = await self.cache.async_get_cache(key=cache_key)
            except Exception:
                continue

            if not cached_model_id or not isinstance(cached_model_id, str):
                continue

            deployment = self._find_deployment_by_model_id(
                healthy_deployments=typed_healthy_deployments,
                model_id=cached_model_id,
            )
            if deployment is not None:
                verbose_router_logger.debug(
                    "EncryptedContentAffinityCheck: item_id=%s pinning -> deployment=%s",
                    item_id,
                    cached_model_id,
                )
                request_kwargs[
                    "_encrypted_content_affinity_pinned"
                ] = True
                return [deployment]

            verbose_router_logger.error(
                "EncryptedContentAffinityCheck: cached deployment=%s for item_id=%s "
                "not found in healthy_deployments",
                cached_model_id,
                item_id,
            )

        return typed_healthy_deployments
