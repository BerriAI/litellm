import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Final

import litellm
from litellm._logging import verbose_router_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.sensitive_data_masker import mask_sensitive_structure
from litellm.router_utils.add_retry_fallback_headers import (
    add_fallback_headers_to_response,
    get_fallback_error_info,
)
from litellm.router_utils.batch_utils import _get_router_metadata_variable_name
from litellm.router_utils.common_utils import resolve_model_group_alias
from litellm.router_utils.cooldown_handlers import (
    _first_present,  # pyright: ignore[reportPrivateUsage] - shared internal helper, used across router_utils
    _set_cooldown_deployments,  # pyright: ignore[reportPrivateUsage] - shared helper, used across router_utils
    cast_exception_status_to_int,
    is_advisor_orchestration_failure,
)
from litellm.router_utils.router_callbacks.track_deployment_metrics import (
    increment_deployment_failures_for_current_minute,
)
from litellm.types.router import LiteLLMParamsTypedDict

if TYPE_CHECKING:
    from litellm.router import Router as _Router

    LitellmRouter = _Router
else:
    LitellmRouter = Any

# Status codes a generic API call's caller-supplied resource id can trigger on its own
# (e.g. a nonexistent file/batch/thread id), independent of the selected deployment's health.
_REQUEST_SCOPED_STATUS_CODES: Final = frozenset((404,))
_ROUTER_METADATA_BUCKETS: Final = ("metadata", "litellm_metadata")
_TEAM_ID_METADATA_KEY: Final = "user_api_key_team_id"
_API_KEY_AUTH_METADATA_KEY: Final = "user_api_key_auth"


@dataclass(frozen=True, slots=True)
class AuthenticatedMetadataContext:
    """Proxy-authenticated metadata that fallback overrides must not replace."""

    team_source_bucket: str | None = None
    api_key_auth_source_bucket: str | None = None
    user_api_key_auth: Any = None
    has_user_api_key_auth: bool = False


def get_authenticated_team_context(
    request_kwargs: Mapping[str, Any],
) -> tuple[str | None, AuthenticatedMetadataContext]:
    """Return authenticated team and API-key context before fallback overrides."""
    authenticated_team_id: str | None = None
    team_source_bucket: str | None = None
    api_key_auth_source_bucket: str | None = None
    user_api_key_auth: Any = None
    has_user_api_key_auth = False

    for bucket_name in _ROUTER_METADATA_BUCKETS:
        bucket = request_kwargs.get(bucket_name)
        if not isinstance(bucket, Mapping):
            continue
        if authenticated_team_id is None:
            team_id = bucket.get(_TEAM_ID_METADATA_KEY)
            if isinstance(team_id, str):
                authenticated_team_id = team_id
                team_source_bucket = bucket_name
        if not has_user_api_key_auth and _API_KEY_AUTH_METADATA_KEY in bucket:
            user_api_key_auth = bucket.get(_API_KEY_AUTH_METADATA_KEY)
            api_key_auth_source_bucket = bucket_name
            has_user_api_key_auth = True

    return authenticated_team_id, AuthenticatedMetadataContext(
        team_source_bucket=team_source_bucket,
        api_key_auth_source_bucket=api_key_auth_source_bucket,
        user_api_key_auth=user_api_key_auth,
        has_user_api_key_auth=has_user_api_key_auth,
    )


def preserve_authenticated_team_context(
    request_kwargs: dict[str, Any],
    authenticated_team_id: str | None,
    source_bucket: AuthenticatedMetadataContext | str | None,
) -> None:
    """Keep proxy-authenticated team/API-key metadata authoritative across fallbacks."""
    context = (
        source_bucket
        if isinstance(source_bucket, AuthenticatedMetadataContext)
        else AuthenticatedMetadataContext(team_source_bucket=source_bucket)
    )

    for bucket_name in _ROUTER_METADATA_BUCKETS:
        bucket = request_kwargs.get(bucket_name)
        if not isinstance(bucket, Mapping):
            continue
        updated_bucket = dict(bucket)
        if authenticated_team_id is None:
            updated_bucket.pop(_TEAM_ID_METADATA_KEY, None)
        else:
            updated_bucket[_TEAM_ID_METADATA_KEY] = authenticated_team_id
        if context.has_user_api_key_auth:
            updated_bucket[_API_KEY_AUTH_METADATA_KEY] = context.user_api_key_auth
        else:
            updated_bucket.pop(_API_KEY_AUTH_METADATA_KEY, None)
        request_kwargs[bucket_name] = updated_bucket

    if authenticated_team_id is not None:
        authoritative_bucket = (
            context.team_source_bucket
            if context.team_source_bucket in _ROUTER_METADATA_BUCKETS
            else "metadata"
        )
        bucket = request_kwargs.get(authoritative_bucket)
        updated_bucket = dict(bucket) if isinstance(bucket, Mapping) else {}
        updated_bucket[_TEAM_ID_METADATA_KEY] = authenticated_team_id
        request_kwargs[authoritative_bucket] = updated_bucket

    if context.has_user_api_key_auth:
        authoritative_bucket = (
            context.api_key_auth_source_bucket
            if context.api_key_auth_source_bucket in _ROUTER_METADATA_BUCKETS
            else "metadata"
        )
        bucket = request_kwargs.get(authoritative_bucket)
        updated_bucket = dict(bucket) if isinstance(bucket, Mapping) else {}
        updated_bucket[_API_KEY_AUTH_METADATA_KEY] = context.user_api_key_auth
        request_kwargs[authoritative_bucket] = updated_bucket


def _trigger_cooldown_for_failed_deployment(
    litellm_router: LitellmRouter,
    kwargs: Mapping[str, Any],
    exception: Exception,
) -> None:
    """
    Trigger cooldown for a failed fallback deployment.

    In the fallback path the normal failure-callback cooldown is skipped because the
    Logging object sets has_logged_async_failure=True after the first failure and
    blocks all subsequent failure callbacks. This helper ensures every failed
    fallback deployment is evaluated for cooldown regardless.
    """
    try:
        if is_advisor_orchestration_failure(exception):
            verbose_router_logger.debug(
                "Not triggering cooldown for fallback deployment: failure originated "
                "from advisor orchestration, not the selected deployment."
            )
            return

        exception_status: Final[str | int] = getattr(exception, "status_code", "")

        if (
            kwargs.get("original_generic_function") is not None
            and cast_exception_status_to_int(exception_status) in _REQUEST_SCOPED_STATUS_CODES
        ):
            verbose_router_logger.debug(
                "Not triggering cooldown for fallback deployment: status %s on a generic API "
                "call is caller-attributable, not a deployment health signal.",
                exception_status,
            )
            return

        if kwargs.get("client_side_timeout") and cast_exception_status_to_int(exception_status) == 408:
            verbose_router_logger.debug(
                "Not triggering cooldown for fallback deployment: a caller-supplied "
                "x-litellm-timeout caused this 408, not deployment health."
            )
            return

        deployment_id: Final[str | None] = getattr(exception, "failed_deployment_id", None)

        if deployment_id is None:
            verbose_router_logger.debug("Cannot trigger cooldown for fallback: no failed_deployment_id on exception")
            return

        deployment_dict: Final = litellm_router.get_model_info(id=deployment_id)
        deployment_cooldown: Final = (
            _first_present(
                deployment_dict.get("model_info"), deployment_dict.get("litellm_params"), key="cooldown_time"
            )
            if deployment_dict is not None
            else None
        )
        exception_headers: Final = litellm.litellm_core_utils.exception_mapping_utils._get_response_headers(
            original_exception=exception
        )
        _get_retry_after: Final = (
            litellm.utils._get_retry_after_from_exception_header  # pyright: ignore[reportPrivateUsage] - as router.py
        )
        header_cooldown: Final = (
            _get_retry_after(response_headers=exception_headers) if exception_headers is not None else None
        )
        time_to_cooldown: Final = (
            deployment_cooldown
            if deployment_cooldown is not None and deployment_cooldown >= 0
            else (
                header_cooldown
                if header_cooldown is not None and header_cooldown >= 0
                else litellm_router.cooldown_time
            )
        )

        increment_deployment_failures_for_current_minute(
            litellm_router_instance=litellm_router,
            deployment_id=deployment_id,
        )
        _set_cooldown_deployments(
            litellm_router_instance=litellm_router,
            exception_status=exception_status,
            original_exception=exception,
            deployment=deployment_id,
            time_to_cooldown=time_to_cooldown,
        )

        verbose_router_logger.debug("Triggered cooldown for fallback deployment %s", deployment_id)
    except Exception as e:  # noqa: BLE001 - best-effort cooldown trigger must never break the fallback response itself
        verbose_router_logger.debug("Error triggering cooldown for fallback deployment: %s", e)


def fallback_attempt_key(fallback_target: object) -> str | None:
    """
    Identity of one fallback attempt, so the same attempt is never made twice per request.

    A bare model group name and a `{"model": name}` entry describe the same attempt. An
    entry carrying anything else describes a different one and keeps its own identity: a
    client-side fallback list overrides request params such as `messages`, and the router
    re-targets the group that just failed by attaching `_target_order` or
    `_excluded_deployment_ids` to select a different set of deployments inside it. The
    payload is hashed rather than kept, so a large `messages` override does not make the
    request hold a second copy of itself.

    Returns None for a shape with no usable identity, which is never skipped.
    """
    if isinstance(fallback_target, str):
        return fallback_target
    if not isinstance(fallback_target, dict):
        return None
    model: Final = fallback_target.get("model")
    if tuple(fallback_target) == ("model",) and isinstance(model, str):
        return model
    serialized: Final = json.dumps(fallback_target, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


@dataclass(slots=True)
class AttemptedFallbackTargets:
    """The fallback attempts a single request has already made."""

    keys: frozenset[str] = frozenset()

    def __contains__(self, key: str) -> bool:
        return key in self.keys

    def record(self, key: str) -> None:
        self.keys = self.keys | frozenset((key,))


def _check_stripped_model_group(model_group: str, fallback_key: str) -> bool:
    for provider in litellm.provider_list:
        if isinstance(provider, Enum):
            _provider = provider.value
        else:
            _provider = provider
        if model_group.startswith(f"{_provider}/"):
            stripped_model_group = model_group.replace(f"{_provider}/", "")
            if stripped_model_group == fallback_key:
                return True
    return False


def get_fallback_model_group(fallbacks: list[Any], model_group: str) -> tuple[list[str] | None, int | None]:
    generic_fallback_idx: int | None = None
    stripped_model_fallback: list[str] | None = None
    fallback_model_group: list[str] | None = None
    for idx, item in enumerate(fallbacks):
        if isinstance(item, dict):
            if list(item.keys())[0] == model_group:
                fallback_model_group = item[model_group]
                break
            elif _check_stripped_model_group(model_group=model_group, fallback_key=list(item.keys())[0]):
                stripped_model_fallback = item[list(item.keys())[0]]
            elif list(item.keys())[0] == "*":
                generic_fallback_idx = idx
        elif isinstance(item, str):
            fallback_model_group = [item]
    if fallback_model_group is None:
        if stripped_model_fallback is not None:
            fallback_model_group = stripped_model_fallback
        elif generic_fallback_idx is not None:
            fallback_model_group = fallbacks[generic_fallback_idx]["*"]

    return fallback_model_group, generic_fallback_idx


PROVIDER_SCOPED_RESOURCE_KEYS: Final = ("input_file_id", "training_file")
PROVIDER_SCOPED_CREATION_FUNCTION_NAMES: Final = frozenset({"_acreate_file"})


def _get_fallback_target_model_group(fallback_entry: str | Mapping[str, object]) -> str | None:
    if isinstance(fallback_entry, str):
        return fallback_entry
    target: Final = fallback_entry.get("model")
    return target if isinstance(target, str) else None


def references_provider_scoped_resource(kwargs: Mapping[str, object]) -> bool:
    return any(kwargs.get(key) for key in PROVIDER_SCOPED_RESOURCE_KEYS)


def creates_provider_scoped_resource(kwargs: Mapping[str, object]) -> bool:
    return getattr(kwargs.get("original_function"), "__name__", None) in PROVIDER_SCOPED_CREATION_FUNCTION_NAMES


async def run_async_fallback(
    *args: tuple[Any],
    litellm_router: LitellmRouter,
    fallback_model_group: list[str],
    original_model_group: str,
    original_exception: Exception,
    max_fallbacks: int,
    fallback_depth: int,
    include_fallback_errors: bool = False,
    **kwargs,
) -> Any:
    if fallback_depth >= max_fallbacks:
        raise original_exception

    error_from_fallbacks = original_exception
    fallback_errors = (get_fallback_error_info(original_exception),)
    metadata_variable_name: Final = _get_router_metadata_variable_name(
        function_name=getattr(kwargs.get("original_function"), "__name__", None)
    )
    authenticated_team_id, authenticated_team_bucket = get_authenticated_team_context(kwargs)
    same_model_group_only: Final = references_provider_scoped_resource(kwargs) or creates_provider_scoped_resource(
        kwargs
    )
    alias_map: Final = getattr(litellm_router, "model_group_alias", None)
    canonical_original_model_group: Final = (
        resolve_model_group_alias(alias_map, original_model_group) or original_model_group
    )
    carried_targets: Final = kwargs.get("attempted_targets")
    attempted: Final = (
        carried_targets if isinstance(carried_targets, AttemptedFallbackTargets) else AttemptedFallbackTargets()
    )
    attempted.record(original_model_group)

    for mg in fallback_model_group:
        target_model_group: Final = _get_fallback_target_model_group(mg)
        canonical_target_model_group: Final = (
            resolve_model_group_alias(alias_map, target_model_group) or target_model_group
            if isinstance(target_model_group, str)
            else None
        )
        if isinstance(mg, str) and canonical_target_model_group == canonical_original_model_group:
            continue
        if same_model_group_only and canonical_target_model_group != canonical_original_model_group:
            verbose_router_logger.info(
                "Skipping fallback to model_group = %s: request is pinned to model_group = %s by its uploaded file",
                mask_sensitive_structure(mg),
                original_model_group,
            )
            continue
        attempt_key = fallback_attempt_key(mg)
        if attempt_key is not None:
            if attempt_key in attempted:
                verbose_router_logger.info(
                    "Skipping fallback to model_group = %s, already attempted for this request",
                    mask_sensitive_structure(mg),
                )
                continue
            attempted.record(attempt_key)
        try:
            # Deployment exclusions belong to the fallback target that just failed.
            # Keep them through that target's retries, then clear them only when
            # advancing to a distinct trusted fallback target.
            kwargs.pop("_excluded_deployment_ids", None)
            kwargs = litellm_router.log_retry(kwargs=kwargs, e=original_exception)
            verbose_router_logger.info("Falling back to model_group = %s", mask_sensitive_structure(mg))
            if isinstance(mg, str):
                kwargs["model"] = mg
            elif isinstance(mg, dict):
                kwargs.update(mg)
            preserve_authenticated_team_context(
                request_kwargs=kwargs,
                authenticated_team_id=authenticated_team_id,
                source_bucket=authenticated_team_bucket,
            )
            fallback_depth = fallback_depth + 1
            kwargs[metadata_variable_name] = {
                "original_model_group": original_model_group,
                **(kwargs.get(metadata_variable_name) or {}),
                "model_group": kwargs.get("model", None),
                "attempted_fallbacks": fallback_depth,
            }
            kwargs["fallback_depth"] = fallback_depth
            kwargs["max_fallbacks"] = max_fallbacks
            kwargs["attempted_targets"] = attempted
            if include_fallback_errors:
                kwargs["include_fallback_errors"] = include_fallback_errors
            response = await litellm_router.async_function_with_fallbacks(*args, **kwargs)
            verbose_router_logger.info("Successful fallback b/w models.")
            response = add_fallback_headers_to_response(
                response=response,
                attempted_fallbacks=fallback_depth,
                fallback_errors=(list(fallback_errors) if include_fallback_errors else None),
            )
            await log_success_fallback_event(
                original_model_group=original_model_group,
                kwargs=kwargs,
                original_exception=original_exception,
            )
            return response
        except Exception as e:
            error_from_fallbacks = e
            fallback_errors = fallback_errors + (get_fallback_error_info(e),)
            await log_failure_fallback_event(
                original_model_group=original_model_group,
                kwargs=kwargs,
                original_exception=original_exception,
            )
            logging_obj = kwargs.get("litellm_logging_obj")
            if logging_obj is not None and logging_obj.model_call_details.get("has_logged_async_failure", False):
                _trigger_cooldown_for_failed_deployment(
                    litellm_router=litellm_router,
                    kwargs=kwargs,
                    exception=e,
                )
    raise error_from_fallbacks


async def log_success_fallback_event(original_model_group: str, kwargs: dict, original_exception: Exception):
    custom_loggers: Final = litellm.logging_callback_manager.get_custom_loggers_for_type(CustomLogger)
    for _callback_custom_logger in custom_loggers:
        try:
            await _callback_custom_logger.log_success_fallback_event(
                original_model_group=original_model_group,
                kwargs=kwargs,
                original_exception=original_exception,
            )
        except Exception as e:
            verbose_router_logger.error("Error in log_success_fallback_event: %s", e)


async def log_failure_fallback_event(original_model_group: str, kwargs: dict, original_exception: Exception):
    custom_loggers: Final = litellm.logging_callback_manager.get_custom_loggers_for_type(CustomLogger)
    for _callback_custom_logger in custom_loggers:
        try:
            await _callback_custom_logger.log_failure_fallback_event(
                original_model_group=original_model_group,
                kwargs=kwargs,
                original_exception=original_exception,
            )
        except Exception as e:
            verbose_router_logger.error("Error in log_failure_fallback_event: %s", e)


def _is_non_standard_fallback_target(item: Any) -> bool:
    if isinstance(item, str):
        return True
    if not isinstance(item, dict):
        return False
    return "model" in item


def _is_unambiguous_direct_fallback_dict(item: Any) -> bool:
    if not isinstance(item, dict) or "model" not in item:
        return False
    model = item.get("model")
    if not isinstance(model, list):
        return True
    return any(key != "model" and not isinstance(value, list) for key, value in item.items())


def _check_non_standard_fallback_format(fallbacks: list[Any] | None) -> bool:
    """Check whether ``fallbacks`` is a direct ordered list of fallback targets."""
    if fallbacks is None or not isinstance(fallbacks, list) or len(fallbacks) == 0:
        return False
    if not all(_is_non_standard_fallback_target(item) for item in fallbacks):
        return False
    if any(isinstance(item, str) for item in fallbacks):
        return True
    return any(_is_unambiguous_direct_fallback_dict(item) for item in fallbacks)


def run_non_standard_fallback_format(fallbacks: list[str] | list[dict[str, Any]], model_group: str):
    pass
