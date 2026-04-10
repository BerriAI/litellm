"""
Aliyun SLS (Simple Log Service) logging integration.

Configuration via litellm.aliyun_sls_callback_params or environment variables:
  aliyun_sls_region            - Region ID, e.g. cn-hangzhou
  aliyun_sls_project           - SLS project name
  aliyun_sls_logstore          - SLS logstore name
  aliyun_sls_access_key_id     - Alibaba Cloud AccessKey ID
  aliyun_sls_access_key_secret - Alibaba Cloud AccessKey Secret
  aliyun_sls_endpoint          - Custom endpoint (optional), overrides region-derived endpoint
"""

import asyncio
import time
from typing import Any, Dict, Optional

from aliyun.log import LogClient, LogItem, PutLogsRequest

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import StandardLoggingPayload


class AliyunSLSLogger(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        params: Dict[str, Any] = {}
        if litellm.aliyun_sls_callback_params is not None:
            for key, value in litellm.aliyun_sls_callback_params.items():
                if isinstance(value, str) and value.startswith("os.environ/"):
                    params[key] = litellm.get_secret(value)
                else:
                    params[key] = value

        region = params.get("aliyun_sls_region") or ""
        self.project = params.get("aliyun_sls_project") or ""
        self.logstore = params.get("aliyun_sls_logstore") or ""
        access_key_id = params.get("aliyun_sls_access_key_id") or ""
        access_key_secret = params.get("aliyun_sls_access_key_secret") or ""
        custom_endpoint = params.get("aliyun_sls_endpoint") or ""

        if not region and not custom_endpoint:
            raise ValueError("aliyun_sls_region is required")
        if not self.project:
            raise ValueError("aliyun_sls_project is required")
        if not self.logstore:
            raise ValueError("aliyun_sls_logstore is required")

        endpoint = custom_endpoint or f"{region}.log.aliyuncs.com"
        self.client = LogClient(endpoint, access_key_id, access_key_secret)
        verbose_logger.debug(
            "AliyunSLSLogger initialized: endpoint=%s project=%s logstore=%s",
            endpoint,
            self.project,
            self.logstore,
        )

    def _put_logs(self, contents: list) -> None:
        log_item = LogItem(timestamp=int(time.time()), contents=contents)
        request = PutLogsRequest(
            project=self.project,
            logstore=self.logstore,
            topic="",
            source="litellm",
            logitems=[log_item],
        )
        self.client.put_logs(request)

    def _build_contents(
        self,
        kwargs: Dict[str, Any],
        _response_obj: Any,
        _start_time: Any,
        _end_time: Any,
        status: str,
    ) -> list:
        payload: Optional[StandardLoggingPayload] = kwargs.get(
            "standard_logging_object"
        )
        if payload is None:
            return [("status", status)]

        contents = [
            ("status", status),
            ("model", str(payload.get("model") or "")),
            ("call_type", str(payload.get("call_type") or "")),
            ("response_id", str(payload.get("id") or "")),
            ("api_base", str(payload.get("api_base") or "")),
            ("response_time_ms", str(payload.get("response_time") or "")),
            ("prompt_tokens", str(payload.get("prompt_tokens") or 0)),
            ("completion_tokens", str(payload.get("completion_tokens") or 0)),
            ("total_tokens", str(payload.get("total_tokens") or 0)),
            ("total_cost", str(payload.get("response_cost") or 0)),
            (
                "user",
                str(payload.get("metadata", {}).get("user_api_key_user_id") or ""),
            ),
            (
                "team_id",
                str(payload.get("metadata", {}).get("user_api_key_team_id") or ""),
            ),
            (
                "key_alias",
                str(payload.get("metadata", {}).get("user_api_key_alias") or ""),
            ),
        ]

        error_str = payload.get("error_str")
        if error_str:
            contents.append(("error", str(error_str)))

        return contents

    async def async_log_success_event(
        self, kwargs: Dict[str, Any], response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        try:
            contents = self._build_contents(
                kwargs, response_obj, start_time, end_time, "success"
            )
            await asyncio.get_running_loop().run_in_executor(
                None, self._put_logs, contents
            )
        except Exception as e:
            verbose_logger.exception(
                "AliyunSLSLogger: failed to log success event: %s", e
            )

    async def async_log_failure_event(
        self, kwargs: Dict[str, Any], response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        try:
            contents = self._build_contents(
                kwargs, response_obj, start_time, end_time, "failure"
            )
            await asyncio.get_running_loop().run_in_executor(
                None, self._put_logs, contents
            )
        except Exception as e:
            verbose_logger.exception(
                "AliyunSLSLogger: failed to log failure event: %s", e
            )
