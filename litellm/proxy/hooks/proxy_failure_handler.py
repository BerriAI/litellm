"""
Runs when LLM Exceptions occur on LiteLLM Proxy
"""

import copy
import json
import uuid

import litellm
from litellm.proxy._types import LiteLLM_ErrorLogs


async def _PROXY_failure_handler(
    kwargs,  # kwargs to completion
    completion_response: litellm.ModelResponse,  # response from completion
    start_time=None,
    end_time=None,  # start/end time for completion
):
    """
    Async Failure Handler - runs when LLM Exceptions occur on LiteLLM Proxy.
    This function logs the errors to the Prisma DB

    Can be disabled by setting the following on proxy_config.yaml:
    ```yaml
    general_settings:
      disable_error_logs: True
    ```

    """
    from litellm._logging import verbose_proxy_logger
    from litellm.proxy.proxy_server import general_settings, prisma_client

    if general_settings.get("disable_error_logs") is True:
        return

    if prisma_client is not None:
        verbose_proxy_logger.debug(
            "inside _PROXY_failure_handler kwargs=", extra=kwargs
        )

        _exception = kwargs.get("exception")
        _exception_type = _exception.__class__.__name__
        _model = kwargs.get("model", None)

        _optional_params = kwargs.get("optional_params", {})
        _optional_params = copy.deepcopy(_optional_params)

        for k, v in _optional_params.items():
            v = str(v)
            v = v[:100]

        _status_code = "500"
        try:
            _status_code = str(_exception.status_code)
        except Exception:
            # Don't let this fail logging the exception to the dB
            pass

        _litellm_params = kwargs.get("litellm_params", {}) or {}
        _metadata = _litellm_params.get("metadata", {}) or {}
        _model_id = _metadata.get("model_info", {}).get("id", "")
        _model_group = _metadata.get("model_group", "")
        api_base = litellm.get_api_base(model=_model, optional_params=_litellm_params)
        _exception_string = str(_exception)

        error_log = LiteLLM_ErrorLogs(
            request_id=str(uuid.uuid4()),
            model_group=_model_group,
            model_id=_model_id,
            litellm_model_name=kwargs.get("model"),
            request_kwargs=_optional_params,
            api_base=api_base,
            exception_type=_exception_type,
            status_code=_status_code,
            exception_string=_exception_string,
            startTime=kwargs.get("start_time"),
            endTime=kwargs.get("end_time"),
        )

        error_log_dict = error_log.model_dump()
        error_log_dict["request_kwargs"] = json.dumps(error_log_dict["request_kwargs"])

        await prisma_client.db.litellm_errorlogs.create(
            data=error_log_dict  # type: ignore
        )

    pass
