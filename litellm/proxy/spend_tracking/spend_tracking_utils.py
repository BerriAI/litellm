import json
import secrets
from datetime import datetime as dt
from typing import Optional, cast

from pydantic import BaseModel

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import SpendLogsMetadata, SpendLogsPayload
from litellm.proxy.utils import PrismaClient, hash_token


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


def get_logging_payload(
    kwargs, response_obj, start_time, end_time, end_user_id: Optional[str]
) -> SpendLogsPayload:

    from litellm.proxy.proxy_server import general_settings, master_key

    verbose_proxy_logger.debug(
        f"SpendTable: get_logging_payload - kwargs: {kwargs}\n\n"
    )

    if kwargs is None:
        kwargs = {}
    if response_obj is None or (
        not isinstance(response_obj, BaseModel) and not isinstance(response_obj, dict)
    ):
        response_obj = {}
    # standardize this function to be used across, s3, dynamoDB, langfuse logging
    litellm_params = kwargs.get("litellm_params", {})
    metadata = (
        litellm_params.get("metadata", {}) or {}
    )  # if litellm_params['metadata'] == None
    completion_start_time = kwargs.get("completion_start_time", end_time)
    call_type = kwargs.get("call_type")
    cache_hit = kwargs.get("cache_hit", False)
    usage = cast(dict, response_obj).get("usage", None) or {}
    if isinstance(usage, litellm.Usage):
        usage = dict(usage)
    id = cast(dict, response_obj).get("id") or kwargs.get("litellm_call_id")
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

    _model_id = metadata.get("model_info", {}).get("id", "")
    _model_group = metadata.get("model_group", "")

    request_tags = (
        json.dumps(metadata.get("tags", []))
        if isinstance(metadata.get("tags", []), list)
        else "[]"
    )

    # clean up litellm metadata
    clean_metadata = SpendLogsMetadata(
        user_api_key=None,
        user_api_key_alias=None,
        user_api_key_team_id=None,
        user_api_key_user_id=None,
        user_api_key_team_alias=None,
        spend_logs_metadata=None,
        requester_ip_address=None,
        additional_usage_values=None,
    )
    if isinstance(metadata, dict):
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
            startTime=start_time,
            endTime=end_time,
            completionStartTime=completion_start_time,
            model=kwargs.get("model", "") or "",
            user=kwargs.get("litellm_params", {})
            .get("metadata", {})
            .get("user_api_key_user_id", "")
            or "",
            team_id=kwargs.get("litellm_params", {})
            .get("metadata", {})
            .get("user_api_key_team_id", "")
            or "",
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
        )

        verbose_proxy_logger.debug(
            "SpendTable: created payload - payload: %s\n\n", payload
        )

        return payload
    except Exception as e:
        verbose_proxy_logger.exception(
            "Error creating spendlogs object - {}".format(str(e))
        )
        raise e


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
