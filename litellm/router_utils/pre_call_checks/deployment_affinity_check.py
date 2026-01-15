"""
Unified deployment affinity (session stickiness) for the Router.

Features (independently enable-able):
1. Responses API continuity: when a `previous_response_id` is provided, route to the
   deployment that generated the original response (highest priority).
2. User-key affinity: map a user key -> deployment id for a TTL and re-use that
   deployment for subsequent requests to the same model group.

This is designed to support "implicit prompt caching" scenarios (no explicit cache_control),
where routing to a consistent deployment is still beneficial.
"""

import hashlib
from typing import Any, Dict, List, Optional, cast

from typing_extensions import TypedDict

from litellm._logging import verbose_router_logger
from litellm.caching.dual_cache import DualCache
from litellm.integrations.custom_logger import CustomLogger, Span
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import CallTypes


class DeploymentAffinityCacheValue(TypedDict):
    model_id: str


class DeploymentAffinityCheck(CustomLogger):
    """
    Router deployment affinity callback.

    NOTE: This is a Router-only callback intended to be wired through
    `Router(optional_pre_call_checks=[...])`.
    """

    CACHE_KEY_PREFIX = "deployment_affinity:v1"

    def __init__(
        self,
        cache: DualCache,
        ttl_seconds: int,
        enable_user_key_affinity: bool,
        enable_responses_api_affinity: bool,
    ):
        self.cache = cache
        self.ttl_seconds = ttl_seconds
        self.enable_user_key_affinity = enable_user_key_affinity
        self.enable_responses_api_affinity = enable_responses_api_affinity

    @staticmethod
    def _hash_user_key(user_key: str) -> str:
        return hashlib.sha256(user_key.encode("utf-8")).hexdigest()

    @staticmethod
    def _shorten_for_logs(value: str, keep: int = 8) -> str:
        if len(value) <= keep:
            return value
        return f"{value[:keep]}..."

    @classmethod
    def get_affinity_cache_key(cls, model_group: str, user_key: str) -> str:
        hashed_user_key = cls._hash_user_key(user_key=user_key)
        return f"{cls.CACHE_KEY_PREFIX}:{model_group}:{hashed_user_key}"

    @staticmethod
    def _get_user_key_from_metadata_dict(metadata: dict) -> Optional[str]:
        user_key = metadata.get("user_api_key_hash") or metadata.get("user_api_key")
        if user_key is None:
            return None
        return str(user_key)

    @staticmethod
    def _get_user_key_from_request_kwargs(request_kwargs: dict) -> Optional[str]:
        """
        Extract a stable user key from request kwargs.

        Primary source (proxy): `metadata.user_api_key_hash` / `metadata.user_api_key`
        Fallback (SDK): `user`
        """
        # 1. Check metadata (Proxy usage)
        metadata = request_kwargs.get("litellm_metadata") or request_kwargs.get("metadata")
        if isinstance(metadata, dict):
            user_key = DeploymentAffinityCheck._get_user_key_from_metadata_dict(
                metadata=metadata
            )
            if user_key is not None:
                return user_key

        # 2. Check top-level 'user' parameter (SDK usage)
        user_key = request_kwargs.get("user")
        if user_key is not None:
            return str(user_key)

        return None

    @staticmethod
    def _find_deployment_by_model_id(
        healthy_deployments: List[dict], model_id: str
    ) -> Optional[dict]:
        for deployment in healthy_deployments:
            deployment_model_id = deployment.get("model_info", {}).get("id")
            if deployment_model_id is not None and str(deployment_model_id) == str(
                model_id
            ):
                return deployment
        return None

    async def async_filter_deployments(
        self,
        model: str,
        healthy_deployments: List,
        messages: Optional[List[AllMessageValues]],
        request_kwargs: Optional[dict] = None,
        parent_otel_span: Optional[Span] = None,
    ) -> List[dict]:
        """
        Optionally filter healthy deployments based on:
        1. `previous_response_id` (Responses API continuity) [highest priority]
        2. cached user-key deployment affinity
        """
        request_kwargs = request_kwargs or {}

        # 1) Responses API continuity (high priority)
        if self.enable_responses_api_affinity:
            previous_response_id = request_kwargs.get("previous_response_id")
            if previous_response_id is not None:
                responses_model_id = ResponsesAPIRequestUtils.get_model_id_from_response_id(
                    str(previous_response_id)
                )
                if responses_model_id is not None:
                    deployment = self._find_deployment_by_model_id(
                        healthy_deployments=cast(List[dict], healthy_deployments),
                        model_id=responses_model_id,
                    )
                    if deployment is not None:
                        verbose_router_logger.debug(
                            "DeploymentAffinityCheck: previous_response_id pinning -> deployment=%s",
                            responses_model_id,
                        )
                        return [deployment]

        # 2) User key -> deployment affinity
        if not self.enable_user_key_affinity:
            return cast(List[dict], healthy_deployments)

        user_key = self._get_user_key_from_request_kwargs(request_kwargs=request_kwargs)
        if user_key is None:
            return cast(List[dict], healthy_deployments)

        cache_key = self.get_affinity_cache_key(model_group=model, user_key=user_key)
        cache_result = await self.cache.async_get_cache(key=cache_key)

        model_id: Optional[str] = None
        if isinstance(cache_result, dict):
            model_id = cast(Optional[str], cache_result.get("model_id"))
        elif isinstance(cache_result, str):
            # Backwards / safety: allow raw string values.
            model_id = cache_result

        if not model_id:
            return cast(List[dict], healthy_deployments)

        deployment = self._find_deployment_by_model_id(
            healthy_deployments=cast(List[dict], healthy_deployments),
            model_id=model_id,
        )
        if deployment is None:
            verbose_router_logger.debug(
                "DeploymentAffinityCheck: pinned deployment=%s not found in healthy_deployments",
                model_id,
            )
            return cast(List[dict], healthy_deployments)

        verbose_router_logger.debug(
            "DeploymentAffinityCheck: user-key affinity hit -> deployment=%s user_key=%s",
            model_id,
            self._shorten_for_logs(user_key),
        )
        return [deployment]

    async def async_pre_call_deployment_hook(
        self, kwargs: Dict[str, Any], call_type: Optional[CallTypes]
    ) -> Optional[dict]:
        """
        Persist/update the user-key -> deployment mapping for this request.

        Why pre-call?
        - LiteLLM runs async success callbacks via a background logging worker for performance.
        - We want affinity to be immediately available for subsequent requests.
        """
        if not self.enable_user_key_affinity:
            return None

        user_key = self._get_user_key_from_request_kwargs(request_kwargs=kwargs)
        if user_key is None:
            return None

        metadata = kwargs.get("litellm_metadata") or kwargs.get("metadata") or {}
        if not isinstance(metadata, dict):
            return None

        model_group = metadata.get("model_group")
        if not model_group:
            return None

        model_info = kwargs.get("model_info") or metadata.get("model_info") or {}
        if not isinstance(model_info, dict):
            return None

        model_id = model_info.get("id")
        if not model_id:
            return None

        cache_key = self.get_affinity_cache_key(
            model_group=str(model_group), user_key=user_key
        )
        try:
            await self.cache.async_set_cache(
                cache_key,
                DeploymentAffinityCacheValue(model_id=str(model_id)),
                ttl=self.ttl_seconds,
            )
            verbose_router_logger.debug(
                "DeploymentAffinityCheck: set affinity mapping model_group=%s deployment=%s ttl=%s user_key=%s",
                model_group,
                model_id,
                self.ttl_seconds,
                self._shorten_for_logs(user_key),
            )
        except Exception as e:
            # Non-blocking: affinity is a best-effort optimization.
            verbose_router_logger.debug(
                "DeploymentAffinityCheck: failed to set cache key=%s: %s",
                cache_key,
                e,
            )

        return None
