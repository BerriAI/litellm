import json
import os
import secrets
import traceback
from typing import Optional

from pydantic import BaseModel

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import SpendLogsMetadata, SpendLogsPayload
from litellm.proxy.utils import hash_token


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
    from pydantic import Json

    from litellm.proxy._types import LiteLLM_SpendLogs
    from litellm.proxy.proxy_server import general_settings, master_key

    verbose_proxy_logger.debug(
        f"SpendTable: get_logging_payload - kwargs: {kwargs}\n\n"
    )

    if kwargs is None:
        kwargs = {}
    if response_obj is None:
        response_obj = {}
    # standardize this function to be used across, s3, dynamoDB, langfuse logging
    litellm_params = kwargs.get("litellm_params", {})
    metadata = (
        litellm_params.get("metadata", {}) or {}
    )  # if litellm_params['metadata'] == None
    completion_start_time = kwargs.get("completion_start_time", end_time)
    call_type = kwargs.get("call_type")
    cache_hit = kwargs.get("cache_hit", False)
    usage = response_obj.get("usage", None) or {}
    if isinstance(usage, litellm.Usage):
        usage = dict(usage)
    id = response_obj.get("id", kwargs.get("litellm_call_id"))
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
