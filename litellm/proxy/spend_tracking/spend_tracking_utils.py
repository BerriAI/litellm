import hashlib
import json
import secrets
from datetime import datetime
from datetime import datetime as dt
from datetime import timezone
from typing import Any, List, Optional, cast

from pydantic import BaseModel

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.core_helpers import get_litellm_metadata_from_kwargs
from litellm.proxy._types import SpendLogsMetadata, SpendLogsPayload
from litellm.proxy.utils import PrismaClient, hash_token
from litellm.types.utils import StandardLoggingPayload
from litellm.utils import get_end_user_id_for_cost_tracking


def _is_master_key(api_key: str, _master_key: Optional[str]) -> bool:
    if _master_key is None:
        return False

    ## string comparison
    is_master_key = secrets.compare_digest(api_key, _master_key)
    if is_master_key:
        return True

    ## hash comparison
    is_master_key = secrets.compare_digest(api_key, hash_token(_master_key))
    if is_master_key:
        return True

    return False


def _get_spend_logs_metadata(
    metadata: Optional[dict], applied_guardrails: Optional[List[str]] = None
) -> SpendLogsMetadata:
    if metadata is None:
        return SpendLogsMetadata(
            user_api_key=None,
            user_api_key_alias=None,
            user_api_key_team_id=None,
            user_api_key_org_id=None,
            user_api_key_user_id=None,
            user_api_key_team_alias=None,
            spend_logs_metadata=None,
            requester_ip_address=None,
            additional_usage_values=None,
            applied_guardrails=None,
            status=None or "success",
            error_information=None,
            proxy_server_request=None,
        )
    verbose_proxy_logger.debug(
        "getting payload for SpendLogs, available keys in metadata: "
        + str(list(metadata.keys()))
    )

    # Filter the metadata dictionary to include only the specified keys
    clean_metadata = SpendLogsMetadata(
        **{  # type: ignore
            key: metadata[key]
            for key in SpendLogsMetadata.__annotations__.keys()
            if key in metadata
        }
    )
    clean_metadata["applied_guardrails"] = applied_guardrails

    return clean_metadata


def generate_hash_from_response(response_obj: Any) -> str:
    """
    Generate a stable hash from a response object.

    Args:
        response_obj: The response object to hash (can be dict, list, etc.)

    Returns:
        A hex string representation of the MD5 hash
    """
    try:
        # Create a stable JSON string of the entire response object
        # Sort keys to ensure consistent ordering
        json_str = json.dumps(response_obj, sort_keys=True)

        # Generate a hash of the response object
        unique_hash = hashlib.md5(json_str.encode()).hexdigest()
        return unique_hash
    except Exception:
        # Return a fallback hash if serialization fails
        return hashlib.md5(str(response_obj).encode()).hexdigest()


def get_spend_logs_id(
    call_type: str, response_obj: dict, kwargs: dict
) -> Optional[str]:
    if call_type == "aretrieve_batch":
        # Generate a hash from the response object
        id: Optional[str] = generate_hash_from_response(response_obj)
    else:
        id = cast(Optional[str], response_obj.get("id")) or cast(
            Optional[str], kwargs.get("litellm_call_id")
        )
    return id


def get_logging_payload(  # noqa: PLR0915
    kwargs, response_obj, start_time, end_time
) -> SpendLogsPayload:
    from litellm.proxy.proxy_server import general_settings, master_key

    if kwargs is None:
        kwargs = {}
    if response_obj is None or (
        not isinstance(response_obj, BaseModel) and not isinstance(response_obj, dict)
    ):
        response_obj = {}
    # standardize this function to be used across, s3, dynamoDB, langfuse logging
    litellm_params = kwargs.get("litellm_params", {})
    metadata = get_litellm_metadata_from_kwargs(kwargs)
    metadata = _add_proxy_server_request_to_metadata(
        metadata=metadata, litellm_params=litellm_params
    )
    completion_start_time = kwargs.get("completion_start_time", end_time)
    call_type = kwargs.get("call_type")
    cache_hit = kwargs.get("cache_hit", False)
    usage = cast(dict, response_obj).get("usage", None) or {}
    if isinstance(usage, litellm.Usage):
        usage = dict(usage)

    if isinstance(response_obj, dict):
        response_obj_dict = response_obj
    elif isinstance(response_obj, BaseModel):
        response_obj_dict = response_obj.model_dump()
    else:
        response_obj_dict = {}

    id = get_spend_logs_id(call_type or "acompletion", response_obj_dict, kwargs)
    standard_logging_payload = cast(
        Optional[StandardLoggingPayload], kwargs.get("standard_logging_object", None)
    )

    end_user_id = get_end_user_id_for_cost_tracking(litellm_params)

    api_key = metadata.get("user_api_key", "")

    if api_key is not None and isinstance(api_key, str):
        if api_key.startswith("sk-"):
            # hash the api_key
            api_key = hash_token(api_key)
        if (
            _is_master_key(api_key=api_key, _master_key=master_key)
            and general_settings.get("disable_adding_master_key_hash_to_db") is True
        ):
            api_key = "litellm_proxy_master_key"  # use a known alias, if the user disabled storing master key in db

    if (
        standard_logging_payload is not None
    ):  # [TODO] migrate completely to sl payload. currently missing pass-through endpoint data
        api_key = (
            api_key
            or standard_logging_payload["metadata"].get("user_api_key_hash")
            or ""
        )
        end_user_id = end_user_id or standard_logging_payload["metadata"].get(
            "user_api_key_end_user_id"
        )
    else:
        api_key = ""
    request_tags = (
        json.dumps(metadata.get("tags", []))
        if isinstance(metadata.get("tags", []), list)
        else "[]"
    )
    if (
        _is_master_key(api_key=api_key, _master_key=master_key)
        and general_settings.get("disable_adding_master_key_hash_to_db") is True
    ):
        api_key = "litellm_proxy_master_key"  # use a known alias, if the user disabled storing master key in db

    _model_id = metadata.get("model_info", {}).get("id", "")
    _model_group = metadata.get("model_group", "")

    # clean up litellm metadata
    clean_metadata = _get_spend_logs_metadata(
        metadata,
        applied_guardrails=(
            standard_logging_payload["metadata"].get("applied_guardrails", None)
            if standard_logging_payload is not None
            else None
        ),
    )

    special_usage_fields = ["completion_tokens", "prompt_tokens", "total_tokens"]
    additional_usage_values = {}
    for k, v in usage.items():
        if k not in special_usage_fields:
            if isinstance(v, BaseModel):
                v = v.model_dump()
            additional_usage_values.update({k: v})
    clean_metadata["additional_usage_values"] = additional_usage_values

    if litellm.cache is not None:
        cache_key = litellm.cache.get_cache_key(**kwargs)
    else:
        cache_key = "Cache OFF"
    if cache_hit is True:
        import time

        id = f"{id}_cache_hit{time.time()}"  # SpendLogs does not allow duplicate request_id
    try:
        payload: SpendLogsPayload = SpendLogsPayload(
            request_id=str(id),
            call_type=call_type or "",
            api_key=str(api_key),
            cache_hit=str(cache_hit),
            startTime=_ensure_datetime_utc(start_time),
            endTime=_ensure_datetime_utc(end_time),
            completionStartTime=_ensure_datetime_utc(completion_start_time),
            model=kwargs.get("model", "") or "",
            user=metadata.get("user_api_key_user_id", "") or "",
            team_id=metadata.get("user_api_key_team_id", "") or "",
            metadata=json.dumps(clean_metadata),
            cache_key=cache_key,
            spend=kwargs.get("response_cost", 0),
            total_tokens=usage.get("total_tokens", 0),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            request_tags=request_tags,
            end_user=end_user_id or "",
            api_base=litellm_params.get("api_base", ""),
            model_group=_model_group,
            model_id=_model_id,
            requester_ip_address=clean_metadata.get("requester_ip_address", None),
            custom_llm_provider=kwargs.get("custom_llm_provider", ""),
            messages=_get_messages_for_spend_logs_payload(
                standard_logging_payload=standard_logging_payload, metadata=metadata
            ),
            response=_get_response_for_spend_logs_payload(standard_logging_payload),
        )

        verbose_proxy_logger.debug(
            "SpendTable: created payload - payload: %s\n\n",
            json.dumps(payload, indent=4, default=str),
        )

        return payload
    except Exception as e:
        verbose_proxy_logger.exception(
            "Error creating spendlogs object - {}".format(str(e))
        )
        raise e


def _ensure_datetime_utc(timestamp: datetime) -> datetime:
    """Helper to ensure datetime is in UTC"""
    timestamp = timestamp.astimezone(timezone.utc)
    return timestamp


async def get_spend_by_team_and_customer(
    start_date: dt,
    end_date: dt,
    team_id: str,
    customer_id: str,
    prisma_client: PrismaClient,
):
    sql_query = """
    WITH SpendByModelApiKey AS (
        SELECT
            date_trunc('day', sl."startTime") AS group_by_day,
            COALESCE(tt.team_alias, 'Unassigned Team') AS team_name,
            sl.end_user AS customer,
            sl.model,
            sl.api_key,
            SUM(sl.spend) AS model_api_spend,
            SUM(sl.total_tokens) AS model_api_tokens
        FROM 
            "LiteLLM_SpendLogs" sl
        LEFT JOIN 
            "LiteLLM_TeamTable" tt 
        ON 
            sl.team_id = tt.team_id
        WHERE
            sl."startTime" BETWEEN $1::date AND $2::date
            AND sl.team_id = $3
            AND sl.end_user = $4
        GROUP BY
            date_trunc('day', sl."startTime"),
            tt.team_alias,
            sl.end_user,
            sl.model,
            sl.api_key
    )
        SELECT
            group_by_day,
            jsonb_agg(jsonb_build_object(
                'team_name', team_name,
                'customer', customer,
                'total_spend', total_spend,
                'metadata', metadata
            )) AS teams_customers
        FROM (
            SELECT
                group_by_day,
                team_name,
                customer,
                SUM(model_api_spend) AS total_spend,
                jsonb_agg(jsonb_build_object(
                    'model', model,
                    'api_key', api_key,
                    'spend', model_api_spend,
                    'total_tokens', model_api_tokens
                )) AS metadata
            FROM 
                SpendByModelApiKey
            GROUP BY
                group_by_day,
                team_name,
                customer
        ) AS aggregated
        GROUP BY
            group_by_day
        ORDER BY
            group_by_day;
    """

    db_response = await prisma_client.db.query_raw(
        sql_query, start_date, end_date, team_id, customer_id
    )
    if db_response is None:
        return []

    return db_response


def _get_messages_for_spend_logs_payload(
    standard_logging_payload: Optional[StandardLoggingPayload],
    metadata: Optional[dict] = None,
) -> str:
    return "{}"


def _add_proxy_server_request_to_metadata(
    metadata: dict,
    litellm_params: dict,
) -> dict:
    """
    Only store if _should_store_prompts_and_responses_in_spend_logs() is True
    """
    if _should_store_prompts_and_responses_in_spend_logs():
        _proxy_server_request = cast(
            Optional[dict], litellm_params.get("proxy_server_request", {})
        )
        if _proxy_server_request is not None:
            _request_body = _proxy_server_request.get("body", {}) or {}
            _request_body_json_str = json.dumps(_request_body, default=str)
            metadata["proxy_server_request"] = _request_body_json_str
    return metadata


def _get_response_for_spend_logs_payload(
    payload: Optional[StandardLoggingPayload],
) -> str:
    if payload is None:
        return "{}"
    if _should_store_prompts_and_responses_in_spend_logs():
        return json.dumps(payload.get("response", {}))
    return "{}"


def _should_store_prompts_and_responses_in_spend_logs() -> bool:
    from litellm.proxy.proxy_server import general_settings

    return general_settings.get("store_prompts_in_spend_logs") is True
