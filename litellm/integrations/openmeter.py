# What is this?
## On Success events log cost to OpenMeter - https://github.com/BerriAI/litellm/issues/1268

import json
import os
from collections.abc import Mapping
from typing import Final, Protocol

import httpx

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.core_helpers import as_str_mapping, require_attributes
from litellm.litellm_core_utils.internal_call_metadata import logged_event_key
from litellm.llms.custom_httpx.http_handler import (
    HTTPHandler,
    get_async_httpx_client,
    httpxSpecialProvider,
)


class _SupportsGet(Protocol):
    def get(self, key: str, default: object = None, /) -> object: ...


def get_utc_datetime():
    import datetime as dt
    from datetime import datetime

    if hasattr(dt, "UTC"):
        return datetime.now(dt.UTC)
    else:
        return datetime.utcnow()


class OpenMeterLogger(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.validate_environment()
        self.async_http_handler = get_async_httpx_client(llm_provider=httpxSpecialProvider.LoggingCallback)
        self.sync_http_handler = HTTPHandler()

    def validate_environment(self):
        """
        Expects
        OPENMETER_API_ENDPOINT,
        OPENMETER_API_KEY,

        in the environment
        """
        missing_keys: Final = []
        if os.getenv("OPENMETER_API_KEY", None) is None:
            missing_keys.append("OPENMETER_API_KEY")

        if len(missing_keys) > 0:
            raise Exception(f"Missing keys={missing_keys} in environment.")

    def _common_logic(self, kwargs: Mapping[str, object], response_obj):
        call_id: Final = response_obj.get("id", kwargs.get("litellm_call_id"))
        dt: Final = get_utc_datetime().isoformat()
        cost: Final = kwargs.get("response_cost", None)
        model: Final = kwargs.get("model")
        usage = {}
        if (
            isinstance(response_obj, litellm.ModelResponse) or isinstance(response_obj, litellm.EmbeddingResponse)
        ) and hasattr(response_obj, "usage"):
            response_usage: Final = require_attributes(response_obj["usage"], _SupportsGet, "get")
            usage = {
                "prompt_tokens": response_usage.get("prompt_tokens", 0),
                "completion_tokens": response_usage.get("completion_tokens", 0),
                "total_tokens": response_usage.get("total_tokens"),
            }

        # OPENMETER_TRUST_REQUEST_USER (default "true"): when set to "false",
        # the request-supplied `user` field is ignored and the subject is
        # resolved solely from the key-bound user_api_key_user_id. Proxies
        # serving multi-tenant traffic enable this to prevent clients from
        # forging attribution by setting `user` in the request body.
        trust_request_user: Final = os.getenv("OPENMETER_TRUST_REQUEST_USER", "true").lower() != "false"
        user_param = kwargs.get("user", None) if trust_request_user else None

        # If no user provided directly, try to get it from token user_id
        if user_param is None:
            # Check if user_id is available from the API key metadata
            litellm_params: Final = as_str_mapping(kwargs.get("litellm_params")) or {}
            metadata: Final = as_str_mapping(litellm_params.get("metadata")) or {}
            user_api_key_user_id: Final = metadata.get("user_api_key_user_id", None)

            if user_api_key_user_id is not None:
                user_param = user_api_key_user_id
            else:
                raise Exception("OpenMeter: user is required")

        # Ensure subject is always a string for OpenMeter API
        subject: Final = str(user_param)

        return {
            "specversion": "1.0",
            "type": os.getenv("OPENMETER_EVENT_TYPE", "litellm_tokens"),
            "id": logged_event_key(kwargs, call_id) if isinstance(call_id, str) else call_id,
            "time": dt,
            "subject": subject,
            "source": "litellm-proxy",
            "data": {"model": model, "cost": cost, **usage},
        }

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        _url = os.getenv("OPENMETER_API_ENDPOINT", "https://openmeter.cloud")
        if _url.endswith("/"):
            _url += "api/v1/events"
        else:
            _url += "/api/v1/events"

        api_key: Final = os.getenv("OPENMETER_API_KEY")

        _data: Final = self._common_logic(kwargs=kwargs, response_obj=response_obj)
        _headers: Final = {
            "Content-Type": "application/cloudevents+json",
            "Authorization": f"Bearer {api_key}",
        }

        try:
            self.sync_http_handler.post(
                url=_url,
                data=json.dumps(_data),
                headers=_headers,
            )
        except httpx.HTTPStatusError as e:
            raise Exception(f"OpenMeter logging error: {e.response.text}")
        except Exception as e:
            raise e

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        _url = os.getenv("OPENMETER_API_ENDPOINT", "https://openmeter.cloud")
        if _url.endswith("/"):
            _url += "api/v1/events"
        else:
            _url += "/api/v1/events"

        api_key: Final = os.getenv("OPENMETER_API_KEY")

        _data: Final = self._common_logic(kwargs=kwargs, response_obj=response_obj)
        _headers: Final = {
            "Content-Type": "application/cloudevents+json",
            "Authorization": f"Bearer {api_key}",
        }

        try:
            await self.async_http_handler.post(
                url=_url,
                data=json.dumps(_data),
                headers=_headers,
            )
        except httpx.HTTPStatusError as e:
            raise Exception(f"OpenMeter logging error: {e.response.text}")
        except Exception as e:
            raise e
