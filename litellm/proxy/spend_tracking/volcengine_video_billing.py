import hashlib
import json
import os
import re
import traceback
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.constants import MAX_OBJECTS_PER_POLL_CYCLE
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.llms.volcengine.videos.transformation import VolcEngineVideoConfig
from litellm.litellm_core_utils.core_helpers import get_litellm_metadata_from_kwargs
from litellm.litellm_core_utils.safe_json_loads import safe_json_loads
from litellm.proxy._types import SpendLogsMetadata, SpendLogsPayload
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders, StandardLoggingPayload
from litellm.types.videos.main import VideoObject
from litellm.types.videos.utils import extract_original_video_id

if TYPE_CHECKING:
    from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
    from litellm.proxy.utils import PrismaClient, ProxyLogging
    from litellm.router import Router
else:
    PrismaClient = Any
    ProxyLogging = Any
    Router = Any
    DBSpendUpdateWriter = Any


VOLCENGINE_VIDEO_CREATE_CALL_TYPES = {
    "avideo_generation",
    "acreate_video",
    "create_video",
}
VOLCENGINE_VIDEO_STATUS_CALL_TYPES = {
    "avideo_status",
    "avideo_retrieve",
    "video_retrieve",
}
VOLCENGINE_VIDEO_CONTENT_CALL_TYPES = {
    "avideo_content",
    "video_content",
}
VOLCENGINE_VIDEO_SUCCESS_CALL_TYPES = (
    VOLCENGINE_VIDEO_CREATE_CALL_TYPES
    | VOLCENGINE_VIDEO_STATUS_CALL_TYPES
    | VOLCENGINE_VIDEO_CONTENT_CALL_TYPES
)
VOLCENGINE_VIDEO_ZERO_COST_CALL_TYPES = VOLCENGINE_VIDEO_SUCCESS_CALL_TYPES
VOLCENGINE_VIDEO_PENDING_STATUSES = {"queued", "processing"}
VOLCENGINE_VIDEO_NO_CHARGE_STATUSES = {"failed", "cancelled", "expired", "deleted"}
VOLCENGINE_VIDEO_COMPLETED_STATUS = "completed"
VOLCENGINE_VIDEO_DEFAULT_PRICING_MODEL = "volcengine/doubao-seedance-2.0"
VOLCENGINE_VIDEO_POLL_INTERVAL_SECONDS = 15
VOLCENGINE_VIDEO_RETRY_INTERVAL_SECONDS = 60
VOLCENGINE_VIDEO_CNY_PER_USD_ENV = "LITELLM_VOLCENGINE_VIDEO_CNY_PER_USD"
VOLCENGINE_VIDEO_DEFAULT_CNY_PER_USD = 7.2
VOLCENGINE_VIDEO_RUNTIME_PRICING_MODELS: Dict[str, Dict[str, Any]] = {
    "volcengine/doubao-seedance-2.0": {
        "litellm_provider": "volcengine",
        "max_input_tokens": 1024,
        "max_output_tokens": 1024,
        "max_tokens": 1024,
        "mode": "video_generation",
        "provider_pricing_currency": "CNY",
        "source": "https://www.volcengine.com/docs/82379/1544106?lang=zh",
        "supported_modalities": ["text", "image", "video", "audio"],
        "supported_output_modalities": ["video"],
        "volcengine_video_output_cost_per_million_tokens_without_input_video": 46.0,
        "volcengine_video_output_cost_per_million_tokens_with_input_video": 28.0,
    },
    "volcengine/doubao-seedance-2.0-fast": {
        "litellm_provider": "volcengine",
        "max_input_tokens": 1024,
        "max_output_tokens": 1024,
        "max_tokens": 1024,
        "mode": "video_generation",
        "provider_pricing_currency": "CNY",
        "source": "https://www.volcengine.com/docs/82379/1544106?lang=zh",
        "supported_modalities": ["text", "image", "video", "audio"],
        "supported_output_modalities": ["video"],
        "volcengine_video_output_cost_per_million_tokens_without_input_video": 37.0,
        "volcengine_video_output_cost_per_million_tokens_with_input_video": 22.0,
    },
}


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _hash_token_if_needed(token: Optional[str]) -> str:
    if token is None:
        return ""
    if token.startswith("sk-"):
        return hashlib.sha256(token.encode()).hexdigest()
    return token


def _normalize_pricing_model(model_name: str) -> str:
    if model_name.startswith("volcengine/"):
        return model_name
    return f"volcengine/{model_name}"


def _candidate_pricing_models(model_name: str) -> List[str]:
    normalized_model = _normalize_pricing_model(model_name)
    candidates = [normalized_model]

    provider, _, raw_model = normalized_model.partition("/")
    versionless_model = re.sub(r"-\d{6,}$", "", raw_model)
    if versionless_model and versionless_model != raw_model:
        candidates.append(f"{provider}/{versionless_model}")

    return candidates


def _has_reference_video(content: Any) -> bool:
    if not isinstance(content, list):
        return False

    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "").lower()
        item_role = str(item.get("role") or "").lower()
        if item_type == "video_url" or item_role == "reference_video":
            return True
        if isinstance(item.get("video_url"), dict):
            return True
    return False


def _parse_request_tags(request_tags: Any) -> List[str]:
    if isinstance(request_tags, list):
        return [str(tag) for tag in request_tags if isinstance(tag, str) and tag]
    if isinstance(request_tags, str):
        parsed = safe_json_loads(request_tags, default=[])
        if isinstance(parsed, list):
            return [str(tag) for tag in parsed if isinstance(tag, str) and tag]
    return []


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _ts_to_datetime(value: Optional[int]) -> Optional[datetime]:
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(int(value), tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return None


def _to_prisma_json(value: Any) -> Any:
    import prisma

    return prisma.Json(value)


def _normalize_non_empty_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    string_value = str(value).strip()
    if not string_value:
        return None
    return string_value


def _get_cny_per_usd_rate() -> float:
    configured_rate = _safe_float(
        os.getenv(VOLCENGINE_VIDEO_CNY_PER_USD_ENV),
        default=VOLCENGINE_VIDEO_DEFAULT_CNY_PER_USD,
    )
    if configured_rate <= 0:
        return VOLCENGINE_VIDEO_DEFAULT_CNY_PER_USD
    return configured_rate


def _convert_provider_spend_to_usd(amount: float, currency: str) -> float:
    normalized_currency = (currency or "USD").upper()
    if normalized_currency == "USD":
        return amount
    if normalized_currency == "CNY":
        return amount / _get_cny_per_usd_rate()

    verbose_proxy_logger.warning(
        "Volcengine video billing unsupported provider currency=%s. Leaving spend unchanged.",
        normalized_currency,
    )
    return amount


class VolcengineVideoBillingManager:
    """
    Accurate async token-based billing for Volcengine video generation.

    Flow:
    1. Create request logs a zero-cost spend log and registers a pending video task.
    2. Status/content requests are always zero-cost request logs.
    3. When the task reaches a terminal status, the finalizer computes provider cost
       from provider-reported usage.total_tokens and updates spend tracking exactly once.
    """

    def __init__(
        self,
        prisma_client: "PrismaClient",
        llm_router: "Router",
        db_spend_update_writer: "DBSpendUpdateWriter",
        proxy_logging_obj: "ProxyLogging",
    ) -> None:
        self.prisma_client = prisma_client
        self.llm_router = llm_router
        self.db_spend_update_writer = db_spend_update_writer
        self.proxy_logging_obj = proxy_logging_obj
        self._video_task_table_unavailable = False
        self._pricing_models_registered = False

    def should_handle_success_event(self, kwargs: dict) -> bool:
        call_type = kwargs.get("call_type")
        if call_type not in VOLCENGINE_VIDEO_SUCCESS_CALL_TYPES:
            return False

        custom_llm_provider = kwargs.get("custom_llm_provider") or (
            kwargs.get("litellm_params", {}) or {}
        ).get("custom_llm_provider")
        return custom_llm_provider == "volcengine"

    async def handle_success_event(
        self,
        kwargs: dict,
        completion_response: Optional[Any],
    ) -> Optional[float]:
        if not self.should_handle_success_event(kwargs):
            return None

        self._force_zero_cost_response(kwargs=kwargs, completion_response=completion_response)

        call_type = kwargs.get("call_type")
        video_response = self._coerce_video_object(completion_response)
        try:
            if (
                call_type in VOLCENGINE_VIDEO_CREATE_CALL_TYPES
                and video_response is not None
            ):
                await self._register_pending_video_task(
                    kwargs=kwargs,
                    completion_response=video_response,
                )
            elif (
                call_type in VOLCENGINE_VIDEO_STATUS_CALL_TYPES
                and video_response is not None
            ):
                await self._reconcile_task_from_video_response(
                    video_id=video_response.id or "",
                    video_response=video_response,
                    kwargs=kwargs,
                )
        except Exception as e:
            verbose_proxy_logger.error(
                "Volcengine video billing success-event hook failed: %s\n%s",
                str(e),
                traceback.format_exc(),
            )
        return 0.0

    async def poll_pending_video_tasks(self) -> None:
        if self.prisma_client is None or self.llm_router is None:
            return

        video_task_table = self._get_video_task_table_model()
        if video_task_table is None:
            return

        now = _now_utc()
        tasks = await video_task_table.find_many(
            where={
                "billing_state": "pending",
                "OR": [
                    {"next_check_at": None},
                    {"next_check_at": {"lte": now}},
                ],
            },
            take=MAX_OBJECTS_PER_POLL_CYCLE,
            order={"created_at": "asc"},
        )

        for task in tasks:
            try:
                await self._poll_single_task(task=task)
            except Exception as e:
                verbose_proxy_logger.error(
                    "Volcengine video billing poll failed for task=%s: %s\n%s",
                    getattr(task, "video_id", None),
                    str(e),
                    traceback.format_exc(),
                )
                await video_task_table.update(
                    where={"video_id": task.video_id},
                    data={
                        "last_error": str(e),
                        "last_checked_at": now,
                        "next_check_at": now
                        + timedelta(seconds=VOLCENGINE_VIDEO_RETRY_INTERVAL_SECONDS),
                        "check_attempts": {"increment": 1},
                    },
                )

    def _force_zero_cost_response(
        self,
        kwargs: dict,
        completion_response: Optional[Any],
    ) -> None:
        kwargs["response_cost"] = 0.0
        standard_logging_object = cast(
            Optional[StandardLoggingPayload], kwargs.get("standard_logging_object")
        )
        if standard_logging_object is not None:
            standard_logging_object["response_cost"] = 0.0

        hidden_params = dict(getattr(completion_response, "_hidden_params", {}) or {})
        if hidden_params:
            hidden_params["response_cost"] = 0.0
            setattr(completion_response, "_hidden_params", hidden_params)

    def _coerce_video_object(
        self,
        completion_response: Optional[Any],
    ) -> Optional[VideoObject]:
        if completion_response is None:
            return None
        if isinstance(completion_response, VideoObject):
            return completion_response

        response_payload: Optional[Dict[str, Any]] = None
        if isinstance(completion_response, dict):
            response_payload = completion_response
        elif hasattr(completion_response, "model_dump"):
            try:
                model_dump = completion_response.model_dump()
                if isinstance(model_dump, dict):
                    response_payload = model_dump
            except Exception:
                response_payload = None
        elif hasattr(completion_response, "dict"):
            try:
                dict_payload = completion_response.dict()
                if isinstance(dict_payload, dict):
                    response_payload = dict_payload
            except Exception:
                response_payload = None

        if not isinstance(response_payload, dict):
            return None

        response_id = response_payload.get("id")
        response_object = response_payload.get("object")
        if response_id is None or response_object != "video":
            return None

        try:
            video_response = VideoObject(**response_payload)
        except Exception:
            return None

        hidden_params = getattr(completion_response, "_hidden_params", None)
        if hidden_params:
            video_response._hidden_params = dict(hidden_params)

        return video_response

    async def _register_pending_video_task(
        self,
        kwargs: dict,
        completion_response: VideoObject,
    ) -> None:
        video_task_table = self._get_video_task_table_model()
        if video_task_table is None:
            return

        video_id = completion_response.id or ""
        if not video_id:
            return

        metadata = get_litellm_metadata_from_kwargs(kwargs=kwargs)
        standard_logging_object = cast(
            Optional[StandardLoggingPayload], kwargs.get("standard_logging_object")
        )

        request_content = await self._get_request_content_for_task_registration(
            kwargs=kwargs,
            completion_response=completion_response,
        )
        has_input_video = _has_reference_video(request_content)

        model_info = cast(dict, metadata.get("model_info", {}) or {})
        pricing_model = self._resolve_pricing_model(model_info=model_info)
        unit_price, pricing_currency = self._resolve_pricing_snapshot(
            pricing_model=pricing_model,
            has_input_video=has_input_video,
        )

        api_key_hash = self._get_api_key_hash(
            metadata=metadata,
            standard_logging_object=standard_logging_object,
        )
        spend_log_identity = await self._get_spend_log_identity(video_id=video_id)
        request_tags = self._get_request_tags(
            metadata=metadata,
            standard_logging_object=standard_logging_object,
        )
        request_tags_json = _to_prisma_json(request_tags)
        task_metadata_json = _to_prisma_json(
            {
                "request_content": request_content,
            }
        )
        now = _now_utc()
        usage = dict(completion_response.usage or {})

        await video_task_table.upsert(
            where={"video_id": video_id},
            data={
                "create": {
                    "video_id": video_id,
                    "provider_task_id": extract_original_video_id(video_id),
                    "api_key": spend_log_identity.get("api_key") or api_key_hash,
                    "user": spend_log_identity.get("user")
                    or metadata.get("user_api_key_user_id")
                    or "",
                    "team_id": spend_log_identity.get("team_id")
                    or metadata.get("user_api_key_team_id")
                    or None,
                    "organization_id": spend_log_identity.get("organization_id")
                    or metadata.get("user_api_key_org_id")
                    or None,
                    "end_user": spend_log_identity.get("end_user")
                    or metadata.get("user_api_key_end_user_id")
                    or None,
                    "custom_llm_provider": kwargs.get("custom_llm_provider") or "",
                    "model": kwargs.get("model") or "",
                    "model_group": metadata.get("model_group") or kwargs.get("model") or "",
                    "model_id": model_info.get("id") or "",
                    "provider_model": completion_response.model or "",
                    "pricing_model": pricing_model,
                    "pricing_currency": pricing_currency,
                    "price_per_million_tokens": unit_price,
                    "has_input_video": has_input_video,
                    "provider_status": completion_response.status or "queued",
                    "duration_seconds": _safe_float(
                        usage.get("duration_seconds"),
                        default=_safe_float(completion_response.seconds, 0.0),
                    )
                    or None,
                    "request_tags": request_tags_json,
                    "metadata": task_metadata_json,
                    "next_check_at": now
                    + timedelta(seconds=VOLCENGINE_VIDEO_POLL_INTERVAL_SECONDS),
                    "last_checked_at": now,
                    "check_attempts": 0,
                },
                "update": {
                    "api_key": spend_log_identity.get("api_key") or api_key_hash,
                    "user": spend_log_identity.get("user")
                    or metadata.get("user_api_key_user_id")
                    or "",
                    "team_id": spend_log_identity.get("team_id")
                    or metadata.get("user_api_key_team_id")
                    or None,
                    "organization_id": spend_log_identity.get("organization_id")
                    or metadata.get("user_api_key_org_id")
                    or None,
                    "end_user": spend_log_identity.get("end_user")
                    or metadata.get("user_api_key_end_user_id")
                    or None,
                    "custom_llm_provider": kwargs.get("custom_llm_provider") or "",
                    "model": kwargs.get("model") or "",
                    "model_group": metadata.get("model_group") or kwargs.get("model") or "",
                    "model_id": model_info.get("id") or "",
                    "provider_model": completion_response.model or "",
                    "pricing_model": pricing_model,
                    "pricing_currency": pricing_currency,
                    "price_per_million_tokens": unit_price,
                    "has_input_video": has_input_video,
                    "provider_status": completion_response.status or "queued",
                    "duration_seconds": _safe_float(
                        usage.get("duration_seconds"),
                        default=_safe_float(completion_response.seconds, 0.0),
                    )
                    or None,
                    "request_tags": request_tags_json,
                    "metadata": task_metadata_json,
                    "next_check_at": now
                    + timedelta(seconds=VOLCENGINE_VIDEO_POLL_INTERVAL_SECONDS),
                    "last_checked_at": now,
                    "last_error": None,
                },
            },
        )

    async def _poll_single_task(self, task: Any) -> None:
        credentials = self._get_volcengine_credentials(task=task)
        if credentials is None:
            raise ValueError(
                f"Could not resolve deployment credentials for model={task.model} model_id={task.model_id}"
            )

        config = VolcEngineVideoConfig()
        provider_model = task.provider_model or task.model or task.model_group or ""
        litellm_params = GenericLiteLLMParams(**credentials)
        headers = config.validate_environment(
            headers={},
            model=provider_model,
            api_key=credentials.get("api_key"),
            litellm_params=litellm_params,
        )
        api_base = config.get_complete_url(
            model=provider_model,
            api_base=credentials.get("api_base"),
            litellm_params=litellm_params.model_dump(exclude_none=True),
        )
        status_url, params = config.transform_video_status_retrieve_request(
            video_id=task.video_id,
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
        )
        async_httpx_client = get_async_httpx_client(llm_provider=LlmProviders.VOLCENGINE)
        response = await async_httpx_client.client.get(
            status_url,
            headers=headers,
            params=params,
        )
        response.raise_for_status()
        video_response = config.transform_video_status_retrieve_response(
            raw_response=response,
            logging_obj=None,
            custom_llm_provider="volcengine",
        )
        await self._reconcile_task_from_video_response(
            video_id=task.video_id,
            video_response=video_response,
        )

    async def _reconcile_task_from_video_response(
        self,
        video_id: str,
        video_response: VideoObject,
        kwargs: Optional[dict] = None,
    ) -> None:
        video_task_table = self._get_video_task_table_model()
        if video_task_table is None:
            return

        if not video_id:
            return

        task = await video_task_table.find_unique(
            where={"video_id": video_id}
        )
        if task is None:
            if kwargs is not None:
                await self._register_pending_video_task(
                    kwargs=kwargs,
                    completion_response=video_response,
                )
                task = await video_task_table.find_unique(where={"video_id": video_id})

        if task is None:
            verbose_proxy_logger.warning(
                "Volcengine video billing: task row missing for video_id=%s",
                video_id,
            )
            return

        provider_status = video_response.status or "queued"
        terminal_completed = provider_status == VOLCENGINE_VIDEO_COMPLETED_STATUS
        terminal_no_charge = provider_status in VOLCENGINE_VIDEO_NO_CHARGE_STATUSES
        now = _now_utc()

        if terminal_completed:
            await self._finalize_completed_task(task=task, video_response=video_response)
            return

        if terminal_no_charge:
            await video_task_table.update(
                where={"video_id": video_id},
                data={
                    "provider_status": provider_status,
                    "billing_state": "no_charge",
                    "completed_at": _ts_to_datetime(video_response.completed_at) or now,
                    "last_checked_at": now,
                    "next_check_at": None,
                    "last_error": None,
                },
            )
            return

        usage = dict(video_response.usage or {})
        await video_task_table.update(
            where={"video_id": video_id},
            data={
                "provider_status": provider_status,
                "duration_seconds": _safe_float(
                    usage.get("duration_seconds"),
                    default=_safe_float(video_response.seconds, 0.0),
                )
                or None,
                "last_checked_at": now,
                "next_check_at": now
                + timedelta(seconds=VOLCENGINE_VIDEO_POLL_INTERVAL_SECONDS),
                "check_attempts": {"increment": 1},
                "last_error": None,
            },
        )

    async def _finalize_completed_task(
        self,
        task: Any,
        video_response: VideoObject,
    ) -> None:
        video_task_table = self._get_video_task_table_model()
        if video_task_table is None:
            return

        if getattr(task, "billing_state", None) in {"billed", "no_charge", "settling"}:
            return

        claim_result = await video_task_table.update_many(
            where={"video_id": task.video_id, "billing_state": "pending"},
            data={
                "billing_state": "settling",
                "provider_status": video_response.status or VOLCENGINE_VIDEO_COMPLETED_STATUS,
                "last_checked_at": _now_utc(),
                "next_check_at": None,
                "last_error": None,
            },
        )
        if claim_result == 0:
            return

        now = _now_utc()
        usage = dict(video_response.usage or {})
        total_tokens = _safe_int(
            usage.get("total_tokens"),
            default=_safe_int(usage.get("completion_tokens")),
        )
        prompt_tokens = _safe_int(usage.get("prompt_tokens"))
        completion_tokens = _safe_int(
            usage.get("completion_tokens"),
            default=total_tokens,
        )
        duration_seconds = _safe_float(
            usage.get("duration_seconds"),
            default=_safe_float(video_response.seconds),
        )

        provider_final_spend = (
            float(task.price_per_million_tokens or 0.0) * float(total_tokens) / 1_000_000.0
        )
        final_spend_usd = _convert_provider_spend_to_usd(
            amount=provider_final_spend,
            currency=task.pricing_currency or "USD",
        )
        delta_spend = max(final_spend_usd - float(task.spend or 0.0), 0.0)
        provider_delta_spend = provider_final_spend
        delta_prompt_tokens = max(prompt_tokens - int(task.prompt_tokens or 0), 0)
        delta_completion_tokens = max(
            completion_tokens - int(task.completion_tokens or 0), 0
        )

        try:
            if delta_spend > 0 or delta_prompt_tokens > 0 or delta_completion_tokens > 0:
                await self._apply_async_billing_delta(
                    task=task,
                    delta_spend=delta_spend,
                    provider_delta_spend=provider_delta_spend,
                    delta_prompt_tokens=delta_prompt_tokens,
                    delta_completion_tokens=delta_completion_tokens,
                    usage=usage,
                )

                await self._upsert_final_spend_log(
                    task=task,
                    final_spend=final_spend_usd,
                    provider_spend_amount=provider_final_spend,
                    total_tokens=total_tokens,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    usage=usage,
                )

            await video_task_table.update(
                where={"video_id": task.video_id},
                data={
                    "provider_status": video_response.status or VOLCENGINE_VIDEO_COMPLETED_STATUS,
                    "billing_state": "billed",
                    "spend": final_spend_usd,
                    "total_tokens": total_tokens,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "duration_seconds": duration_seconds or None,
                    "completed_at": _ts_to_datetime(video_response.completed_at) or now,
                    "billed_at": now,
                    "last_checked_at": now,
                    "next_check_at": None,
                    "check_attempts": {"increment": 1},
                    "last_error": None,
                },
            )
        except Exception as e:
            await video_task_table.update(
                where={"video_id": task.video_id},
                data={
                    "billing_state": "pending",
                    "last_error": str(e),
                    "last_checked_at": now,
                    "next_check_at": now
                    + timedelta(seconds=VOLCENGINE_VIDEO_RETRY_INTERVAL_SECONDS),
                    "check_attempts": {"increment": 1},
                },
            )
            raise

    async def _apply_async_billing_delta(
        self,
        task: Any,
        delta_spend: float,
        provider_delta_spend: float,
        delta_prompt_tokens: int,
        delta_completion_tokens: int,
        usage: Dict[str, Any],
    ) -> None:
        from litellm.proxy.proxy_server import (
            litellm_proxy_budget_name,
            update_cache,
            user_api_key_cache,
        )

        spend_log_identity = await self._get_spend_log_identity(video_id=task.video_id)
        effective_api_key = spend_log_identity.get("api_key") or task.api_key or ""
        effective_user = spend_log_identity.get("user") or task.user or None
        effective_team_id = spend_log_identity.get("team_id") or task.team_id or None
        effective_org_id = (
            spend_log_identity.get("organization_id") or task.organization_id or None
        )
        effective_end_user = spend_log_identity.get("end_user") or task.end_user or None
        request_tags = _parse_request_tags(task.request_tags)
        delta_metadata: SpendLogsMetadata = cast(
            SpendLogsMetadata,
            {
                "usage_object": usage,
                "async_billing_only": True,
                "provider_spend_currency": task.pricing_currency or "CNY",
                "provider_spend_amount": provider_delta_spend,
                "billing_spend_currency": "USD",
                "billing_spend_amount": delta_spend,
                "provider_to_usd_fx_rate": _get_cny_per_usd_rate()
                if (task.pricing_currency or "").upper() == "CNY"
                else None,
                "video_billing_task_id": task.video_id,
            },
        )
        delta_payload: SpendLogsPayload = cast(
            SpendLogsPayload,
            {
                "request_id": task.video_id,
                "call_type": "avideo_generation",
                "api_key": effective_api_key,
                "spend": delta_spend,
                "total_tokens": delta_prompt_tokens + delta_completion_tokens,
                "prompt_tokens": delta_prompt_tokens,
                "completion_tokens": delta_completion_tokens,
                "startTime": task.created_at,
                "endTime": task.created_at,
                "completionStartTime": None,
                "model": task.model or "",
                "model_id": task.model_id or "",
                "model_group": task.model_group or task.model or "",
                "mcp_namespaced_tool_name": None,
                "agent_id": None,
                "api_base": "",
                "user": effective_user or "",
                "metadata": json.dumps(delta_metadata),
                "cache_hit": "False",
                "cache_key": "Cache OFF",
                "request_tags": json.dumps(request_tags),
                "team_id": effective_team_id,
                "organization_id": effective_org_id,
                "end_user": effective_end_user,
                "requester_ip_address": None,
                "custom_llm_provider": task.custom_llm_provider or "volcengine",
                "messages": None,
                "response": None,
                "proxy_server_request": None,
                "session_id": None,
                "request_duration_ms": None,
                "status": "success",
            },
        )

        await self.db_spend_update_writer.apply_async_billing_delta(
            response_cost=delta_spend,
            user_id=effective_user,
            hashed_token=effective_api_key or None,
            team_id=effective_team_id,
            org_id=effective_org_id,
            end_user_id=effective_end_user,
            prisma_client=self.prisma_client,
            user_api_key_cache=user_api_key_cache,
            litellm_proxy_budget_name=litellm_proxy_budget_name,
            payload=delta_payload,
            request_tags=request_tags,
        )

        await update_cache(
            token=effective_api_key or None,
            user_id=effective_user,
            end_user_id=effective_end_user,
            team_id=effective_team_id,
            response_cost=delta_spend,
            parent_otel_span=None,
            tags=request_tags,
        )

    async def _upsert_final_spend_log(
        self,
        task: Any,
        final_spend: float,
        provider_spend_amount: float,
        total_tokens: int,
        prompt_tokens: int,
        completion_tokens: int,
        usage: Dict[str, Any],
    ) -> None:
        existing_spend_log = await self.prisma_client.db.litellm_spendlogs.find_unique(
            where={"request_id": task.video_id}
        )
        metadata = self._build_final_spend_log_metadata(
            existing_metadata=getattr(existing_spend_log, "metadata", None),
            usage=usage,
            pricing_currency=task.pricing_currency or "CNY",
            provider_spend_amount=provider_spend_amount,
            final_spend=final_spend,
            video_task_id=task.video_id,
        )
        metadata_json = _to_prisma_json(metadata)
        request_tags_json = _to_prisma_json(_parse_request_tags(task.request_tags))
        empty_json_object = _to_prisma_json({})

        if existing_spend_log is None:
            create_data = {
                "request_id": task.video_id,
                "call_type": "avideo_generation",
                "api_key": task.api_key or "",
                "spend": final_spend,
                "total_tokens": total_tokens,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "startTime": task.created_at,
                "endTime": task.created_at,
                "model": task.model or "",
                "model_id": task.model_id or "",
                "model_group": task.model_group or task.model or "",
                "custom_llm_provider": task.custom_llm_provider or "volcengine",
                "api_base": "",
                "user": task.user or "",
                "metadata": metadata_json,
                "cache_hit": "False",
                "cache_key": "Cache OFF",
                "request_tags": request_tags_json,
                "team_id": task.team_id,
                "organization_id": task.organization_id,
                "end_user": task.end_user,
                "messages": empty_json_object,
                "response": empty_json_object,
                "proxy_server_request": empty_json_object,
                "status": "success",
            }
            await self.prisma_client.db.litellm_spendlogs.upsert(
                where={"request_id": task.video_id},
                data={
                    "create": create_data,
                    "update": {
                        "spend": final_spend,
                        "total_tokens": total_tokens,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "metadata": metadata_json,
                    },
                },
            )
            return

        await self.prisma_client.db.litellm_spendlogs.update(
            where={"request_id": task.video_id},
            data={
                "spend": final_spend,
                "total_tokens": total_tokens,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "metadata": metadata_json,
            },
        )

    def _build_final_spend_log_metadata(
        self,
        existing_metadata: Any,
        usage: Dict[str, Any],
        pricing_currency: str,
        provider_spend_amount: float,
        final_spend: float,
        video_task_id: str,
    ) -> Dict[str, Any]:
        if isinstance(existing_metadata, dict):
            metadata_dict = dict(existing_metadata)
        else:
            metadata_dict = safe_json_loads(existing_metadata, default={})
            if not isinstance(metadata_dict, dict):
                metadata_dict = {}
        metadata_dict["usage_object"] = usage
        metadata_dict["provider_spend_currency"] = pricing_currency
        metadata_dict["provider_spend_amount"] = provider_spend_amount
        metadata_dict["billing_spend_currency"] = "USD"
        metadata_dict["billing_spend_amount"] = final_spend
        if pricing_currency.upper() == "CNY":
            metadata_dict["provider_to_usd_fx_rate"] = _get_cny_per_usd_rate()
        metadata_dict["video_billing_task_id"] = video_task_id
        return metadata_dict

    async def _get_request_content_for_task_registration(
        self,
        kwargs: dict,
        completion_response: VideoObject,
    ) -> Optional[List[Dict[str, Any]]]:
        request_content = cast(
            Optional[List[Dict[str, Any]]],
            (getattr(completion_response, "_hidden_params", {}) or {}).get(
                "request_content"
            ),
        )
        if request_content is not None:
            return request_content

        request_content = self._build_request_content_from_proxy_server_request(
            kwargs.get("proxy_server_request")
        )
        if request_content is not None:
            return request_content

        return await self._get_request_content_from_existing_spend_log(
            video_id=completion_response.id or ""
        )

    async def _get_request_content_from_existing_spend_log(
        self,
        video_id: str,
    ) -> Optional[List[Dict[str, Any]]]:
        existing_spend_log = await self._get_existing_spend_log(video_id=video_id)
        if existing_spend_log is None:
            return None

        return self._build_request_content_from_proxy_server_request(
            getattr(existing_spend_log, "proxy_server_request", None)
        )

    async def _get_existing_spend_log(self, video_id: str) -> Optional[Any]:
        if not video_id or self.prisma_client is None:
            return None

        return await self.prisma_client.db.litellm_spendlogs.find_unique(
            where={"request_id": video_id}
        )

    async def _get_spend_log_identity(self, video_id: str) -> Dict[str, Optional[str]]:
        existing_spend_log = await self._get_existing_spend_log(video_id=video_id)
        if existing_spend_log is None:
            return {
                "api_key": None,
                "user": None,
                "team_id": None,
                "organization_id": None,
                "end_user": None,
            }

        return {
            "api_key": _normalize_non_empty_str(
                getattr(existing_spend_log, "api_key", None)
            ),
            "user": _normalize_non_empty_str(getattr(existing_spend_log, "user", None)),
            "team_id": _normalize_non_empty_str(
                getattr(existing_spend_log, "team_id", None)
            ),
            "organization_id": _normalize_non_empty_str(
                getattr(existing_spend_log, "organization_id", None)
            ),
            "end_user": _normalize_non_empty_str(
                getattr(existing_spend_log, "end_user", None)
            ),
        }

    def _build_request_content_from_proxy_server_request(
        self,
        proxy_server_request: Any,
    ) -> Optional[List[Dict[str, Any]]]:
        if isinstance(proxy_server_request, str):
            proxy_server_request = safe_json_loads(proxy_server_request, default={})
        if not isinstance(proxy_server_request, dict):
            return None

        prompt = proxy_server_request.get("prompt") or ""
        provided_content = proxy_server_request.get("content")
        input_reference = proxy_server_request.get("input_reference")

        try:
            return VolcEngineVideoConfig()._build_content_list(
                prompt=str(prompt),
                provided_content=deepcopy(provided_content),
                input_reference=deepcopy(input_reference),
            )
        except Exception:
            return None

    def _get_api_key_hash(
        self,
        metadata: dict,
        standard_logging_object: Optional[StandardLoggingPayload],
    ) -> str:
        if standard_logging_object is not None:
            api_key_hash = (
                (standard_logging_object.get("metadata", {}) or {}).get(
                    "user_api_key_hash"
                )
                or ""
            )
            if api_key_hash:
                return str(api_key_hash)
        return _hash_token_if_needed(cast(Optional[str], metadata.get("user_api_key")))

    def _get_request_tags(
        self,
        metadata: dict,
        standard_logging_object: Optional[StandardLoggingPayload],
    ) -> List[str]:
        if standard_logging_object is not None and standard_logging_object.get(
            "request_tags"
        ) is not None:
            return _parse_request_tags(standard_logging_object.get("request_tags"))
        return _parse_request_tags(metadata.get("tags"))

    def _resolve_pricing_model(self, model_info: dict) -> str:
        pricing_model = (
            model_info.get("provider_pricing_model")
            or model_info.get("base_model")
            or VOLCENGINE_VIDEO_DEFAULT_PRICING_MODEL
        )
        normalized_model = _normalize_pricing_model(str(pricing_model))
        if normalized_model == VOLCENGINE_VIDEO_DEFAULT_PRICING_MODEL and not (
            model_info.get("provider_pricing_model") or model_info.get("base_model")
        ):
            verbose_proxy_logger.warning(
                "Volcengine video billing falling back to default pricing model=%s. "
                "Set model_info.provider_pricing_model or model_info.base_model for exact endpoint pricing.",
                VOLCENGINE_VIDEO_DEFAULT_PRICING_MODEL,
            )
        return normalized_model

    def _resolve_pricing_snapshot(
        self,
        pricing_model: str,
        has_input_video: bool,
    ) -> Tuple[float, str]:
        self._ensure_runtime_pricing_models_registered()

        pricing_entry = None
        pricing_key = None
        for candidate in _candidate_pricing_models(pricing_model):
            pricing_entry = litellm.model_cost.get(candidate)
            if pricing_entry is not None:
                pricing_key = candidate
                break

        if pricing_entry is None or pricing_key is None:
            raise ValueError(f"No pricing config found for Volcengine video model={pricing_model}")

        price_key = (
            "volcengine_video_output_cost_per_million_tokens_with_input_video"
            if has_input_video
            else "volcengine_video_output_cost_per_million_tokens_without_input_video"
        )
        unit_price = pricing_entry.get(price_key)
        if unit_price is None:
            raise ValueError(
                f"Missing pricing key={price_key} for Volcengine video model={pricing_key}"
            )
        pricing_currency = pricing_entry.get("provider_pricing_currency", "CNY")
        return float(unit_price), str(pricing_currency)

    def _ensure_runtime_pricing_models_registered(self) -> None:
        if self._pricing_models_registered:
            return

        should_register = False
        for model_name, model_info in VOLCENGINE_VIDEO_RUNTIME_PRICING_MODELS.items():
            existing_model_info = litellm.model_cost.get(model_name) or {}
            if not all(
                key in existing_model_info
                for key in (
                    "provider_pricing_currency",
                    "volcengine_video_output_cost_per_million_tokens_without_input_video",
                    "volcengine_video_output_cost_per_million_tokens_with_input_video",
                )
            ):
                should_register = True
                break

        if should_register:
            litellm.register_model(model_cost=VOLCENGINE_VIDEO_RUNTIME_PRICING_MODELS)
            verbose_proxy_logger.info(
                "Registered runtime pricing overrides for Volcengine video billing"
            )

        self._pricing_models_registered = True

    def _get_video_task_table_model(self) -> Optional[Any]:
        if self._video_task_table_unavailable or self.prisma_client is None:
            return None

        try:
            return getattr(self.prisma_client.db, "litellm_videotasktable")
        except AttributeError:
            self._video_task_table_unavailable = True
            verbose_proxy_logger.warning(
                "Volcengine video billing disabled: Prisma client is missing "
                "litellm_videotasktable. Run `poetry run prisma generate`, apply the "
                "latest proxy migrations, and restart the proxy."
            )
            return None

    def _get_volcengine_credentials(self, task: Any) -> Optional[Dict[str, Any]]:
        for candidate in (
            task.model_id,
            task.model_group,
            task.model,
            task.provider_model,
        ):
            if not candidate:
                continue
            credentials = self.llm_router.get_deployment_credentials_with_provider(
                candidate
            )
            if credentials and credentials.get("custom_llm_provider") == "volcengine":
                return credentials
        return None
