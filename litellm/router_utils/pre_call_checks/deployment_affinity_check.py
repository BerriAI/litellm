"""
Unified deployment affinity (session stickiness) for the Router.

Features (independently enable-able):
1. Responses API continuity: when a `previous_response_id` is provided, route to the
   deployment that generated the original response (highest priority).
2. API-key affinity: map an API key hash -> deployment id for a TTL and re-use that
   deployment for subsequent requests to the same router deployment model name
   (alias-safe, aligns to `model_map_information.model_map_key`).

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
        enable_session_id_affinity: bool = False,
    ):
        super().__init__()
        self.cache = cache
        self.ttl_seconds = ttl_seconds
        self.enable_user_key_affinity = enable_user_key_affinity
        self.enable_responses_api_affinity = enable_responses_api_affinity
        self.enable_session_id_affinity = enable_session_id_affinity

    @staticmethod
    def _looks_like_sha256_hex(value: str) -> bool:
        if len(value) != 64:
            return False
        try:
            int(value, 16)
        except ValueError:
            return False
        return True

    @staticmethod
    def _hash_user_key(user_key: str) -> str:
        """
        Hash user identifiers before storing them in cache keys.

        This avoids putting raw API keys / user identifiers into Redis keys (and therefore
        into logs/metrics), while keeping the cache key stable and a fixed length.
        """
        # If the proxy already provides a stable SHA-256 (e.g. `metadata.user_api_key_hash`),
        # keep it as-is to avoid double-hashing and to make correlation/debugging possible.
        if DeploymentAffinityCheck._looks_like_sha256_hex(user_key):
            return user_key.lower()

        return hashlib.sha256(user_key.encode("utf-8")).hexdigest()

    @staticmethod
    def _get_model_map_key_from_litellm_model_name(
        litellm_model_name: str,
    ) -> Optional[str]:
        """
        Best-effort derivation of a stable "model map key" for affinity scoping.

        The intent is to align with `standard_logging_payload.model_map_information.model_map_key`,
        which is typically the base model identifier (stable across deployments/endpoints).

        Notes:
        - When the model name is in "provider/model" format, the provider prefix is stripped.
        - For Azure, the string after "azure/" is commonly an *Azure deployment name*, which may
          differ across instances. If `base_model` is not explicitly set, we skip deriving a
          model-map key from the model string to avoid generating unstable keys.
        """
        if not litellm_model_name:
            return None

        if "/" not in litellm_model_name:
            return litellm_model_name

        provider_prefix, remainder = litellm_model_name.split("/", 1)
        if provider_prefix == "azure":
            return None

        return remainder

    @staticmethod
    def _get_model_map_key_from_deployment(deployment: dict) -> Optional[str]:
        """
        Derive a stable model-map key from a router deployment dict.

        Primary source: `deployment.model_name` (Router's canonical group name after
        alias resolution). This is stable across provider-specific deployments (e.g.,
        Azure/Vertex/Bedrock for the same logical model) and aligns with
        `model_map_information.model_map_key` in standard logging.

        Prefer `base_model` when available (important for Azure), otherwise fall back to
        parsing `litellm_params.model`.
        """
        model_name = deployment.get("model_name")
        if isinstance(model_name, str) and model_name:
            return model_name

        model_info = deployment.get("model_info")
        if isinstance(model_info, dict):
            base_model = model_info.get("base_model")
            if isinstance(base_model, str) and base_model:
                return base_model

        litellm_params = deployment.get("litellm_params")
        if isinstance(litellm_params, dict):
            base_model = litellm_params.get("base_model")
            if isinstance(base_model, str) and base_model:
                return base_model
            litellm_model_name = litellm_params.get("model")
            if isinstance(litellm_model_name, str) and litellm_model_name:
                return (
                    DeploymentAffinityCheck._get_model_map_key_from_litellm_model_name(
                        litellm_model_name
                    )
                )

        return None

    @staticmethod
    def _get_stable_model_map_key_from_deployments(
        healthy_deployments: List[dict],
    ) -> Optional[str]:
        """
        Only use model-map key scoping when it is stable across the deployment set.

        This prevents accidentally keying on per-deployment identifiers like Azure deployment
        names (when `base_model` is not configured).
        """
        if not healthy_deployments:
            return None

        keys: List[str] = []
        for deployment in healthy_deployments:
            key = DeploymentAffinityCheck._get_model_map_key_from_deployment(deployment)
            if key is None:
                return None
            keys.append(key)

        unique_keys = set(keys)
        if len(unique_keys) != 1:
            return None
        return keys[0]

    @staticmethod
    def _shorten_for_logs(value: str, keep: int = 8) -> str:
        if len(value) <= keep:
            return value
        return f"{value[:keep]}..."

    @classmethod
    def get_affinity_cache_key(cls, model_group: str, user_key: str) -> str:
        hashed_user_key = cls._hash_user_key(user_key=user_key)
        return f"{cls.CACHE_KEY_PREFIX}:{model_group}:{hashed_user_key}"

    @classmethod
    def get_session_affinity_cache_key(cls, model_group: str, session_id: str) -> str:
        return f"{cls.CACHE_KEY_PREFIX}:session:{model_group}:{session_id}"

    @staticmethod
    def _get_user_key_from_metadata_dict(metadata: dict) -> Optional[str]:
        # NOTE: affinity is keyed on the *API key hash* provided by the proxy (not the
        # OpenAI `user` parameter, which is an end-user identifier).
        user_key = metadata.get("user_api_key_hash")
        if user_key is None:
            return None
        return str(user_key)

    @staticmethod
    def _get_session_id_from_metadata_dict(metadata: dict) -> Optional[str]:
        session_id = metadata.get("session_id")
        if session_id is None:
            return None
        return str(session_id)

    @staticmethod
    def _iter_metadata_dicts(request_kwargs: dict) -> List[dict]:
        """
        Return all metadata dicts available on the request.

        Depending on the endpoint, Router may populate `metadata` or `litellm_metadata`.
        Users may also send one or both, so we check both (rather than using `or`).
        """
        metadata_dicts: List[dict] = []
        for key in ("litellm_metadata", "metadata"):
            md = request_kwargs.get(key)
            if isinstance(md, dict):
                metadata_dicts.append(md)
        return metadata_dicts

    @staticmethod
    def _get_user_key_from_request_kwargs(request_kwargs: dict) -> Optional[str]:
        """
        Extract a stable affinity key from request kwargs.

        Source (proxy): `metadata.user_api_key_hash`

        Note: the OpenAI `user` parameter is an end-user identifier and is intentionally
        not used for deployment affinity.
        """
        # Check metadata dicts (Proxy usage)
        for metadata in DeploymentAffinityCheck._iter_metadata_dicts(request_kwargs):
            user_key = DeploymentAffinityCheck._get_user_key_from_metadata_dict(
                metadata=metadata
            )
            if user_key is not None:
                return user_key

        return None

    @staticmethod
    def _get_session_id_from_request_kwargs(request_kwargs: dict) -> Optional[str]:
        for metadata in DeploymentAffinityCheck._iter_metadata_dicts(request_kwargs):
            session_id = DeploymentAffinityCheck._get_session_id_from_metadata_dict(
                metadata=metadata
            )
            if session_id is not None:
                return session_id
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
        2. cached API-key deployment affinity
        """
        request_kwargs = request_kwargs or {}
        typed_healthy_deployments = cast(List[dict], healthy_deployments)

        # 1) Responses API continuity (high priority)
        if self.enable_responses_api_affinity:
            previous_response_id = request_kwargs.get("previous_response_id")
            if previous_response_id is not None:
                responses_model_id = (
                    ResponsesAPIRequestUtils.get_model_id_from_response_id(
                        str(previous_response_id)
                    )
                )
                if responses_model_id is not None:
                    deployment = self._find_deployment_by_model_id(
                        healthy_deployments=typed_healthy_deployments,
                        model_id=responses_model_id,
                    )
                    if deployment is not None:
                        verbose_router_logger.debug(
                            "DeploymentAffinityCheck: previous_response_id pinning -> deployment=%s",
                            responses_model_id,
                        )
                        return [deployment]

        stable_model_map_key = self._get_stable_model_map_key_from_deployments(
            healthy_deployments=typed_healthy_deployments
        )
        if stable_model_map_key is None:
            return typed_healthy_deployments

        # 2) Session-id -> deployment affinity
        if self.enable_session_id_affinity:
            session_id = self._get_session_id_from_request_kwargs(
                request_kwargs=request_kwargs
            )
            if session_id is not None:
                session_cache_key = self.get_session_affinity_cache_key(
                    model_group=stable_model_map_key, session_id=session_id
                )
                session_cache_result = await self.cache.async_get_cache(
                    key=session_cache_key
                )

                session_model_id: Optional[str] = None
                if isinstance(session_cache_result, dict):
                    session_model_id = cast(
                        Optional[str], session_cache_result.get("model_id")
                    )
                elif isinstance(session_cache_result, str):
                    session_model_id = session_cache_result

                if session_model_id:
                    session_deployment = self._find_deployment_by_model_id(
                        healthy_deployments=typed_healthy_deployments,
                        model_id=session_model_id,
                    )
                    if session_deployment is not None:
                        verbose_router_logger.debug(
                            "DeploymentAffinityCheck: session-id affinity hit -> deployment=%s session_id=%s",
                            session_model_id,
                            session_id,
                        )
                        return [session_deployment]
                    else:
                        verbose_router_logger.debug(
                            "DeploymentAffinityCheck: session-id pinned deployment=%s not found in healthy_deployments",
                            session_model_id,
                        )

        # 3) User key -> deployment affinity
        if not self.enable_user_key_affinity:
            return typed_healthy_deployments

        user_key = self._get_user_key_from_request_kwargs(request_kwargs=request_kwargs)
        if user_key is None:
            return typed_healthy_deployments

        cache_key = self.get_affinity_cache_key(
            model_group=stable_model_map_key, user_key=user_key
        )
        cache_result = await self.cache.async_get_cache(key=cache_key)

        model_id: Optional[str] = None
        if isinstance(cache_result, dict):
            model_id = cast(Optional[str], cache_result.get("model_id"))
        elif isinstance(cache_result, str):
            # Backwards / safety: allow raw string values.
            model_id = cache_result

        if not model_id:
            return typed_healthy_deployments

        deployment = self._find_deployment_by_model_id(
            healthy_deployments=typed_healthy_deployments,
            model_id=model_id,
        )
        if deployment is None:
            verbose_router_logger.debug(
                "DeploymentAffinityCheck: pinned deployment=%s not found in healthy_deployments",
                model_id,
            )
            return typed_healthy_deployments

        verbose_router_logger.debug(
            "DeploymentAffinityCheck: api-key affinity hit -> deployment=%s user_key=%s",
            model_id,
            self._shorten_for_logs(user_key),
        )
        return [deployment]

    async def async_pre_call_deployment_hook(
        self, kwargs: Dict[str, Any], call_type: Optional[CallTypes]
    ) -> Optional[dict]:
        """
        Persist/update the API-key -> deployment mapping for this request.

        Why pre-call?
        - LiteLLM runs async success callbacks via a background logging worker for performance.
        - We want affinity to be immediately available for subsequent requests.
        """
        if not self.enable_user_key_affinity and not self.enable_session_id_affinity:
            return None

        user_key = None
        if self.enable_user_key_affinity:
            user_key = self._get_user_key_from_request_kwargs(request_kwargs=kwargs)

        session_id = None
        if self.enable_session_id_affinity:
            session_id = self._get_session_id_from_request_kwargs(request_kwargs=kwargs)

        if user_key is None and session_id is None:
            return None

        metadata_dicts = self._iter_metadata_dicts(kwargs)

        model_info = kwargs.get("model_info")
        if not isinstance(model_info, dict):
            model_info = None

        if model_info is None:
            for metadata in metadata_dicts:
                maybe_model_info = metadata.get("model_info")
                if isinstance(maybe_model_info, dict):
                    model_info = maybe_model_info
                    break

        if model_info is None:
            # Router sets `model_info` after selecting a deployment. If it's missing, this is
            # likely a non-router call or a call path that doesn't support affinity.
            return None

        model_id = model_info.get("id")
        if not model_id:
            verbose_router_logger.warning(
                "DeploymentAffinityCheck: model_id missing; skipping affinity cache update."
            )
            return None

        # Scope affinity by the Router deployment model name (alias-safe, consistent across
        # heterogeneous providers, and matches standard logging's `model_map_key`).
        deployment_model_name: Optional[str] = None
        for metadata in metadata_dicts:
            maybe_deployment_model_name = metadata.get("deployment_model_name")
            if (
                isinstance(maybe_deployment_model_name, str)
                and maybe_deployment_model_name
            ):
                deployment_model_name = maybe_deployment_model_name
                break

        if not deployment_model_name:
            verbose_router_logger.warning(
                "DeploymentAffinityCheck: deployment_model_name missing; skipping affinity cache update. model_id=%s",
                model_id,
            )
            return None

        if user_key is not None:
            try:
                cache_key = self.get_affinity_cache_key(
                    model_group=deployment_model_name, user_key=user_key
                )
                await self.cache.async_set_cache(
                    cache_key,
                    DeploymentAffinityCacheValue(model_id=str(model_id)),
                    ttl=self.ttl_seconds,
                )

                verbose_router_logger.debug(
                    "DeploymentAffinityCheck: set affinity mapping model_map_key=%s deployment=%s ttl=%s user_key=%s",
                    deployment_model_name,
                    model_id,
                    self.ttl_seconds,
                    self._shorten_for_logs(user_key),
                )
            except Exception as e:
                # Non-blocking: affinity is a best-effort optimization.
                verbose_router_logger.debug(
                    "DeploymentAffinityCheck: failed to set user key affinity cache. model_map_key=%s error=%s",
                    deployment_model_name,
                    e,
                )

        # Also persist Session-ID affinity if enabled and session-id is provided
        if session_id is not None:
            try:
                session_cache_key = self.get_session_affinity_cache_key(
                    model_group=deployment_model_name, session_id=session_id
                )
                await self.cache.async_set_cache(
                    session_cache_key,
                    DeploymentAffinityCacheValue(model_id=str(model_id)),
                    ttl=self.ttl_seconds,
                )
                verbose_router_logger.debug(
                    "DeploymentAffinityCheck: set session affinity mapping model_map_key=%s deployment=%s ttl=%s session_id=%s",
                    deployment_model_name,
                    model_id,
                    self.ttl_seconds,
                    session_id,
                )
            except Exception as e:
                verbose_router_logger.debug(
                    "DeploymentAffinityCheck: failed to set session affinity cache. model_map_key=%s error=%s",
                    deployment_model_name,
                    e,
                )

        return None
