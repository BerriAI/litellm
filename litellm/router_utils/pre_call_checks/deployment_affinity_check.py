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
import json
from collections.abc import Mapping, Sequence
from typing import Any, Final, cast

from typing_extensions import TypedDict

from litellm._logging import verbose_router_logger
from litellm.caching.dual_cache import DualCache
from litellm.constants import SESSION_DEPLOYMENT_AFFINITY_TTL_METADATA_KEY
from litellm.integrations.custom_logger import CustomLogger, Span
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import CallTypes


class DeploymentAffinityCacheValue(TypedDict):
    model_id: str


VALID_MODEL_GROUP_AFFINITY_FLAGS: Final = frozenset(
    {
        "deployment_affinity",
        "responses_api_deployment_check",
        "session_affinity",
        "encrypted_content_affinity",
    }
)


def warn_on_unknown_model_group_affinity_flags(model_group_affinity_config: Mapping[str, Sequence[str]] | None) -> None:
    """`model_group_affinity_config` is one Router-level config consumed by two callbacks:
    DeploymentAffinityCheck acts on three of the flags and EncryptedContentAffinityCheck
    on the fourth, so typo detection lives here at the schema, not inside either consumer.
    """
    if model_group_affinity_config is None:
        return
    for group, flags in model_group_affinity_config.items():
        unknown = set(flags) - VALID_MODEL_GROUP_AFFINITY_FLAGS
        if unknown:
            verbose_router_logger.warning(
                "model_group_affinity_config: unknown flag(s) %s for model group '%s'; will be ignored. Valid flags: %s",
                unknown,
                group,
                VALID_MODEL_GROUP_AFFINITY_FLAGS,
            )


_CLAIM_PIN_SCRIPT: Final = """
local current = redis.call('GET', KEYS[1])
if current == false then
  redis.call('SET', KEYS[1], ARGV[1], 'EX', ARGV[2])
  return ARGV[1]
end
if current == ARGV[1] then
  redis.call('EXPIRE', KEYS[1], ARGV[2])
end
return current
"""


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
        model_group_affinity_config: dict[str, list[str]] | None = None,
    ):
        super().__init__()
        self.cache = cache
        self.ttl_seconds = ttl_seconds
        self.enable_user_key_affinity = enable_user_key_affinity
        self.enable_responses_api_affinity = enable_responses_api_affinity
        self.enable_session_id_affinity = enable_session_id_affinity
        self.model_group_affinity_config: dict[str, list[str]] = model_group_affinity_config or {}

    def _get_effective_flags(self, model_group: str) -> tuple[bool, bool, bool]:
        """
        Return (enable_user_key_affinity, enable_responses_api_affinity, enable_session_id_affinity)
        for the given model group.

        If the model group has an explicit entry in model_group_affinity_config, use it.
        Otherwise fall back to the global instance flags.
        """
        group_checks: Final = self.model_group_affinity_config.get(model_group)
        if group_checks is not None:
            return (
                "deployment_affinity" in group_checks,
                "responses_api_deployment_check" in group_checks,
                "session_affinity" in group_checks,
            )
        return (
            self.enable_user_key_affinity,
            self.enable_responses_api_affinity,
            self.enable_session_id_affinity,
        )

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
    ) -> str | None:
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
    def _get_model_map_key_from_deployment(deployment: dict) -> str | None:
        """
        Derive a stable model-map key from a router deployment dict.

        Primary source: `deployment.model_name` (Router's canonical group name after
        alias resolution). This is stable across provider-specific deployments (e.g.,
        Azure/Vertex/Bedrock for the same logical model) and aligns with
        `model_map_information.model_map_key` in standard logging.

        Prefer `base_model` when available (important for Azure), otherwise fall back to
        parsing `litellm_params.model`.
        """
        model_name: Final = deployment.get("model_name")
        if isinstance(model_name, str) and model_name:
            return model_name

        model_info: Final = deployment.get("model_info")
        if isinstance(model_info, dict):
            base_model = model_info.get("base_model")
            if isinstance(base_model, str) and base_model:
                return base_model

        litellm_params: Final = deployment.get("litellm_params")
        if isinstance(litellm_params, dict):
            base_model = litellm_params.get("base_model")
            if isinstance(base_model, str) and base_model:
                return base_model
            litellm_model_name: Final = litellm_params.get("model")
            if isinstance(litellm_model_name, str) and litellm_model_name:
                return DeploymentAffinityCheck._get_model_map_key_from_litellm_model_name(litellm_model_name)

        return None

    @staticmethod
    def _get_stable_model_map_key_from_deployments(
        healthy_deployments: list[dict],
    ) -> str | None:
        """
        Only use model-map key scoping when it is stable across the deployment set.

        This prevents accidentally keying on per-deployment identifiers like Azure deployment
        names (when `base_model` is not configured).
        """
        if not healthy_deployments:
            return None

        keys: Final[list[str]] = []
        for deployment in healthy_deployments:
            key = DeploymentAffinityCheck._get_model_map_key_from_deployment(deployment)
            if key is None:
                return None
            keys.append(key)

        unique_keys: Final = set(keys)
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
        hashed_user_key: Final = cls._hash_user_key(user_key=user_key)
        return f"{cls.CACHE_KEY_PREFIX}:{model_group}:{hashed_user_key}"

    @classmethod
    def get_session_affinity_cache_key(cls, model_group: str, session_id: str, user_key: str | None) -> str:
        """Session pins are scoped by the caller's hashed API key so two callers reusing
        the same client-supplied session_id cannot read or steer each other's pin.
        `"unscoped"` covers direct Router usage with no authenticated caller, matching
        the complexity router's own session pin key."""
        hashed_user_key: Final = cls._hash_user_key(user_key) if user_key is not None else "unscoped"
        return f"{cls.CACHE_KEY_PREFIX}:session:{model_group}:{hashed_user_key}:{session_id}"

    @staticmethod
    def _get_user_key_from_metadata_dict(metadata: dict) -> str | None:
        # NOTE: affinity is keyed on the *API key hash* provided by the proxy (not the
        # OpenAI `user` parameter, which is an end-user identifier).
        user_key: Final = metadata.get("user_api_key_hash")
        if user_key is None:
            return None
        return str(user_key)

    @staticmethod
    def _get_session_id_from_metadata_dict(metadata: dict) -> str | None:
        session_id: Final = metadata.get("session_id")
        if session_id is None:
            return None
        return str(session_id)

    @staticmethod
    def _iter_metadata_dicts(request_kwargs: dict) -> list[dict]:
        """
        Return all metadata dicts available on the request.

        Depending on the endpoint, Router may populate `metadata` or `litellm_metadata`.
        Users may also send one or both, so we check both (rather than using `or`).
        """
        metadata_dicts: Final[list[dict]] = []
        for key in ("litellm_metadata", "metadata"):
            md = request_kwargs.get(key)
            if isinstance(md, dict):
                metadata_dicts.append(md)
        return metadata_dicts

    @staticmethod
    def _get_user_key_from_request_kwargs(request_kwargs: dict) -> str | None:
        """
        Extract a stable affinity key from request kwargs.

        Source (proxy): `metadata.user_api_key_hash`

        Note: the OpenAI `user` parameter is an end-user identifier and is intentionally
        not used for deployment affinity.
        """
        # Check metadata dicts (Proxy usage)
        for metadata in DeploymentAffinityCheck._iter_metadata_dicts(request_kwargs):
            user_key = DeploymentAffinityCheck._get_user_key_from_metadata_dict(metadata=metadata)
            if user_key is not None:
                return user_key

        return None

    @staticmethod
    def _get_session_id_from_request_kwargs(request_kwargs: dict) -> str | None:
        for metadata in DeploymentAffinityCheck._iter_metadata_dicts(request_kwargs):
            session_id = DeploymentAffinityCheck._get_session_id_from_metadata_dict(metadata=metadata)
            if session_id is not None:
                return session_id
        return None

    @staticmethod
    def _get_marker_session_affinity_ttl(request_kwargs: dict) -> int | None:
        """TTL from the session-affinity marker the Router stamps at pre-routing time
        when an auto-router routed this request with session_affinity enabled.
        Marker presence enables session pinning for this request only; anything that
        is not a positive int is treated as absent."""
        for metadata in DeploymentAffinityCheck._iter_metadata_dicts(request_kwargs):
            ttl = metadata.get(SESSION_DEPLOYMENT_AFFINITY_TTL_METADATA_KEY)
            if isinstance(ttl, int) and not isinstance(ttl, bool) and ttl > 0:
                return ttl
        return None

    @staticmethod
    def _pinned_model_id(stored: object) -> str | None:
        """Deployment id held by a stored pin, for both the dict shape this writes and the
        bare string older writers left behind. None when the value is neither."""
        if isinstance(stored, dict):
            model_id: Final = stored.get("model_id")
            return str(model_id) if model_id is not None else None
        if isinstance(stored, str):
            return stored
        return None

    def _set_local_pin(self, cache_key: str, value: object, ttl_seconds: int) -> None:
        """The one owner of authoritative local pin writes: a plain set keeps a live
        key's original expiry (`allow_ttl_override`), so the entry is replaced to make
        the TTL real. Every local pin write goes through here so the redis-winner sync
        and the pod-local claim can never disagree about expiry again."""
        self.cache.in_memory_cache.delete_cache(cache_key)
        self.cache.in_memory_cache.set_cache(cache_key, value, ttl=ttl_seconds)

    async def _claim_pin(self, cache_key: str, pin_value: DeploymentAffinityCacheValue, ttl_seconds: int) -> str | None:
        """First-writer-wins pin write: store `pin_value` only when the key is absent and
        return the deployment id the key holds afterwards, so a caller learns whether it won
        by comparing against its own id, and None when the stored value is one no reader can
        interpret. Concurrent claimers converge on the
        first write instead of the last. Re-claiming with the stored value refreshes its
        TTL, the same keepalive the complexity router's model pin documents: an active
        session must not lose its pin mid-conversation just because it outlives the
        original write, so `session_affinity_ttl_seconds` bounds idle time, not total
        session length. On Redis one Lua script does the get-or-set-or-refresh
        atomically (same registration seam the rate limiters use) and the in-memory
        tier is synchronized to the winner; without Redis, and whenever Redis is
        unreachable, the pod-local check-and-set below stands in and is atomic because it
        runs synchronously on the event loop. Degrading to a pod-local claim rather than
        propagating the fault is what keeps same-pod stickiness through a Redis blip: the
        caller only logs this result, so an escaping error would leave the session with no
        pin at all and reshuffle every turn for the outage, which is worse than losing
        cross-pod agreement. The redis tier is
        resolved per call because the proxy attaches it after Router construction
        (`Router._update_redis_cache`); the compiled script is cached per event loop
        underneath the registration seam.
        """
        redis_cache: Final = self.cache.redis_cache
        if redis_cache is not None:
            try:
                claim_script: Final = redis_cache.async_register_script(_CLAIM_PIN_SCRIPT)
                raw: Final = await claim_script(keys=(cache_key,), args=(json.dumps(pin_value), int(ttl_seconds)))
                decoded: Final = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                if not isinstance(decoded, str):
                    return pin_value["model_id"]
                try:
                    winner: object = json.loads(decoded)
                except json.JSONDecodeError:
                    winner = decoded
                self._set_local_pin(cache_key=cache_key, value=winner, ttl_seconds=ttl_seconds)
                return self._pinned_model_id(winner)
            except Exception as e:  # noqa: BLE001  # any Redis/Lua failure degrades to the pod-local claim, never unpins
                verbose_router_logger.debug(
                    "DeploymentAffinityCheck: redis pin claim failed, falling back to pod-local claim. error=%s", e
                )

        return self._claim_pin_in_memory(cache_key=cache_key, pin_value=pin_value, ttl_seconds=ttl_seconds)

    def _claim_pin_in_memory(
        self, cache_key: str, pin_value: DeploymentAffinityCacheValue, ttl_seconds: int
    ) -> str | None:
        """Pod-local half of the claim, used when no Redis tier is attached and as the
        fallback when the Redis claim fails. Mirrors the Lua script exactly, including
        the keepalive: re-claiming with the stored value slides the idle window through
        `_set_local_pin`. Both branches stay synchronous, hence atomic on the event
        loop."""
        existing: Final = self.cache.in_memory_cache.get_cache(cache_key)
        if existing is not None:
            existing_model_id: Final = self._pinned_model_id(existing)
            if existing_model_id == pin_value["model_id"]:
                self._set_local_pin(cache_key=cache_key, value=pin_value, ttl_seconds=ttl_seconds)
            return existing_model_id
        self._set_local_pin(cache_key=cache_key, value=pin_value, ttl_seconds=ttl_seconds)
        return pin_value["model_id"]

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

    async def async_filter_deployments(
        self,
        model: str,
        healthy_deployments: list,
        messages: list[AllMessageValues] | None,
        request_kwargs: dict | None = None,
        parent_otel_span: Span | None = None,
    ) -> list[dict]:
        """
        Optionally filter healthy deployments based on:
        1. `previous_response_id` (Responses API continuity) [highest priority]
        2. cached API-key deployment affinity
        """
        request_kwargs = request_kwargs or {}
        typed_healthy_deployments: Final = cast(list[dict], healthy_deployments)

        (
            enable_user_key,
            enable_responses_api,
            enable_session_id,
        ) = self._get_effective_flags(model)

        # 1) Responses API continuity (high priority)
        if enable_responses_api:
            previous_response_id: Final = request_kwargs.get("previous_response_id")
            if previous_response_id is not None:
                responses_model_id = ResponsesAPIRequestUtils.get_model_id_from_response_id(str(previous_response_id))
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

        stable_model_map_key: Final = self._get_stable_model_map_key_from_deployments(
            healthy_deployments=typed_healthy_deployments
        )
        if stable_model_map_key is None:
            return typed_healthy_deployments

        session_affinity_active: Final = (
            enable_session_id or self._get_marker_session_affinity_ttl(request_kwargs=request_kwargs) is not None
        )
        user_key: Final = (
            self._get_user_key_from_request_kwargs(request_kwargs=request_kwargs)
            if (session_affinity_active or enable_user_key)
            else None
        )

        # 2) Session-id -> deployment affinity
        if session_affinity_active:
            session_id: Final = self._get_session_id_from_request_kwargs(request_kwargs=request_kwargs)
            if session_id is not None:
                session_cache_key: Final = self.get_session_affinity_cache_key(
                    model_group=stable_model_map_key, session_id=session_id, user_key=user_key
                )
                session_cache_result: Final = await self.cache.async_get_cache(key=session_cache_key)

                session_model_id: str | None = None
                if isinstance(session_cache_result, dict):
                    session_model_id = cast(str | None, session_cache_result.get("model_id"))
                elif isinstance(session_cache_result, str):
                    session_model_id = session_cache_result

                if session_model_id:
                    session_deployment: Final = self._find_deployment_by_model_id(
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
        if not enable_user_key:
            return typed_healthy_deployments

        if user_key is None:
            return typed_healthy_deployments

        cache_key: Final = self.get_affinity_cache_key(model_group=stable_model_map_key, user_key=user_key)
        cache_result: Final = await self.cache.async_get_cache(key=cache_key)

        model_id: str | None = None
        if isinstance(cache_result, dict):
            model_id = cast(str | None, cache_result.get("model_id"))
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

    async def async_pre_call_deployment_hook(self, kwargs: dict[str, Any], call_type: CallTypes | None) -> dict | None:
        """
        Persist/update the API-key -> deployment mapping for this request.

        Why pre-call?
        - LiteLLM runs async success callbacks via a background logging worker for performance.
        - We want affinity to be immediately available for subsequent requests.
        """
        metadata_dicts: Final = self._iter_metadata_dicts(kwargs)

        # Extract deployment_model_name first — needed for both per-group flag resolution
        # and cache key scoping.
        deployment_model_name: str | None = None
        for metadata in metadata_dicts:
            maybe_deployment_model_name = metadata.get("deployment_model_name")
            if isinstance(maybe_deployment_model_name, str) and maybe_deployment_model_name:
                deployment_model_name = maybe_deployment_model_name
                break

        if not deployment_model_name:
            verbose_router_logger.debug(
                "DeploymentAffinityCheck: deployment_model_name missing in metadata; skipping affinity cache update."
            )
            return None

        # Resolve effective flags for this model group
        (
            enable_user_key,
            _enable_responses_api,
            enable_session_id,
        ) = self._get_effective_flags(deployment_model_name)

        marker_session_ttl: Final = self._get_marker_session_affinity_ttl(request_kwargs=kwargs)
        session_affinity_active: Final = enable_session_id or marker_session_ttl is not None

        if not enable_user_key and not session_affinity_active:
            return None

        user_key: Final = (
            self._get_user_key_from_request_kwargs(request_kwargs=kwargs)
            if (enable_user_key or session_affinity_active)
            else None
        )
        session_id: Final = (
            self._get_session_id_from_request_kwargs(request_kwargs=kwargs) if session_affinity_active else None
        )

        if not ((enable_user_key and user_key is not None) or session_id is not None):
            return None

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

        model_id: Final = model_info.get("id")
        if not model_id:
            verbose_router_logger.warning("DeploymentAffinityCheck: model_id missing; skipping affinity cache update.")
            return None

        pin_value: Final = DeploymentAffinityCacheValue(model_id=str(model_id))

        if enable_user_key and user_key is not None:
            try:
                cache_key: Final = self.get_affinity_cache_key(model_group=deployment_model_name, user_key=user_key)
                claimed_user_pin: Final = await self._claim_pin(
                    cache_key=cache_key,
                    pin_value=pin_value,
                    ttl_seconds=self.ttl_seconds,
                )
                if claimed_user_pin == pin_value["model_id"]:
                    verbose_router_logger.debug(
                        "DeploymentAffinityCheck: set affinity mapping model_map_key=%s deployment=%s ttl=%s user_key=%s",
                        deployment_model_name,
                        model_id,
                        self.ttl_seconds,
                        self._shorten_for_logs(user_key),
                    )
                else:
                    verbose_router_logger.debug(
                        "DeploymentAffinityCheck: affinity pin already claimed model_map_key=%s existing=%s ours=%s",
                        deployment_model_name,
                        claimed_user_pin,
                        model_id,
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
                session_affinity_ttl: Final = marker_session_ttl if marker_session_ttl is not None else self.ttl_seconds
                session_cache_key: Final = self.get_session_affinity_cache_key(
                    model_group=deployment_model_name, session_id=session_id, user_key=user_key
                )
                claimed_session_pin: Final = await self._claim_pin(
                    cache_key=session_cache_key,
                    pin_value=pin_value,
                    ttl_seconds=session_affinity_ttl,
                )
                if claimed_session_pin == pin_value["model_id"]:
                    verbose_router_logger.debug(
                        "DeploymentAffinityCheck: set session affinity mapping model_map_key=%s deployment=%s ttl=%s session_id=%s",
                        deployment_model_name,
                        model_id,
                        session_affinity_ttl,
                        session_id,
                    )
                else:
                    verbose_router_logger.debug(
                        "DeploymentAffinityCheck: session pin already claimed model_map_key=%s existing=%s ours=%s session_id=%s",
                        deployment_model_name,
                        claimed_session_pin,
                        model_id,
                        session_id,
                    )
            except Exception as e:
                verbose_router_logger.debug(
                    "DeploymentAffinityCheck: failed to set session affinity cache. model_map_key=%s error=%s",
                    deployment_model_name,
                    e,
                )

        return None
