# What is this?
## On Success events log cost to OpenMeter - https://github.com/BerriAI/litellm/issues/1268

import json
import os

import httpx

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.custom_httpx.http_handler import (
    HTTPHandler,
    get_async_httpx_client,
    httpxSpecialProvider,
)


def get_utc_datetime():
    import datetime as dt
    from datetime import datetime

    if hasattr(dt, "UTC"):
        return datetime.now(dt.UTC)  # type: ignore
    else:
        return datetime.utcnow()  # type: ignore


class OpenMeterLogger(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.validate_environment()
        self.async_http_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback
        )
        self.sync_http_handler = HTTPHandler()

    def validate_environment(self):
        """
        Expects
        OPENMETER_API_ENDPOINT,
        OPENMETER_API_KEY,

        in the environment
        """
        missing_keys = []
        if os.getenv("OPENMETER_API_KEY", None) is None:
            missing_keys.append("OPENMETER_API_KEY")

        if len(missing_keys) > 0:
            raise Exception("Missing keys={} in environment.".format(missing_keys))

    def _common_logic(self, kwargs: dict, response_obj):
        call_id = response_obj.get("id", kwargs.get("litellm_call_id"))
        dt = get_utc_datetime().isoformat()
        cost = kwargs.get("response_cost", None)
        model = kwargs.get("model")
        usage = {}
        if (
            isinstance(response_obj, litellm.ModelResponse)
            or isinstance(response_obj, litellm.EmbeddingResponse)
        ) and hasattr(response_obj, "usage"):
            usage = {
                "prompt_tokens": response_obj["usage"].get("prompt_tokens", 0),
                "completion_tokens": response_obj["usage"].get("completion_tokens", 0),
                "total_tokens": response_obj["usage"].get("total_tokens"),
            }

        user_param = kwargs.get("user", None)  # end-user passed in via 'user' param
        if user_param is None:
            raise Exception("OpenMeter: user is required")
        
        # Ensure subject is always a string for OpenMeter API
        subject = str(user_param)

        return {
            "specversion": "1.0",
            "type": os.getenv("OPENMETER_EVENT_TYPE", "litellm_tokens"),
            "id": call_id,
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

        api_key = os.getenv("OPENMETER_API_KEY")

        _data = self._common_logic(kwargs=kwargs, response_obj=response_obj)
        _headers = {
            "Content-Type": "application/cloudevents+json",
            "Authorization": "Bearer {}".format(api_key),
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

        api_key = os.getenv("OPENMETER_API_KEY")

        _data = self._common_logic(kwargs=kwargs, response_obj=response_obj)
        _headers = {
            "Content-Type": "application/cloudevents+json",
            "Authorization": "Bearer {}".format(api_key),
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
