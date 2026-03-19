from __future__ import annotations
from uuid import uuid4
import httpx

import os
from typing import TYPE_CHECKING, Any, Literal, Optional, Type
from urllib.parse import urlparse

import requests
from fastapi import HTTPException
from httpx import HTTPStatusError
from requests.auth import HTTPBasicAuth

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.proxy.guardrails.guardrail_hooks.hiddenlayer import (
    HiddenlayerAction,
    HiddenlayerMessages,
)
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel


def is_saas(host: str) -> bool:
    """Checks whether the connection is to the SaaS platform"""

    o = urlparse(host)

    if o.hostname and o.hostname.endswith("hiddenlayer.ai"):
        return True

    return False


def _get_jwt(auth_url, api_id, api_key):
    token_url = f"{auth_url}/oauth2/token?grant_type=client_credentials"

    resp = requests.post(token_url, auth=HTTPBasicAuth(api_id, api_key))

    if not resp.ok:
        raise RuntimeError(
            f"Unable to get authentication credentials for the HiddenLayer API: {resp.status_code}: {resp.text}"
        )

    if "access_token" not in resp.json():
        raise RuntimeError(
            f"Unable to get authentication credentials for the HiddenLayer API - invalid response: {resp.json()}"
        )

    return resp.json()["access_token"]


class HiddenlayerGuardrail(CustomGuardrail):
    """Custom guardrail wrapper for HiddenLayer's safety checks."""

    def __init__(
        self,
        api_id: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        auth_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.hiddenlayer_client_id = api_id or os.getenv("HIDDENLAYER_CLIENT_ID")
        self.hiddenlayer_client_secret = api_key or os.getenv(
            "HIDDENLAYER_CLIENT_SECRET"
        )
        self.api_base = (
            api_base
            or os.getenv("HIDDENLAYER_API_BASE")
            or "https://api.hiddenlayer.ai"
        )
        self.jwt_token = None

        auth_url = (
            auth_url
            or os.getenv("HIDDENLAYER_AUTH_URL")
            or "https://auth.hiddenlayer.ai"
        )

        if is_saas(self.api_base):
            if not self.hiddenlayer_client_id:
                raise RuntimeError(
                    "`api_id` cannot be None when using the SaaS version of HiddenLayer."
                )

            if not self.hiddenlayer_client_secret:
                raise RuntimeError(
                    "`api_key` cannot be None when using the SaaS version of HiddenLayer."
                )

            self.jwt_token = _get_jwt(
                auth_url=auth_url,
                api_id=self.hiddenlayer_client_id,
                api_key=self.hiddenlayer_client_secret,
            )
            self.refresh_jwt_func = lambda: _get_jwt(
                auth_url=auth_url,
                api_id=self.hiddenlayer_client_id,
                api_key=self.hiddenlayer_client_secret,
            )

        self._http_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )
        super().__init__(**kwargs)

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        """Validate (and optionally redact) text via HiddenLayer before/after LLM calls."""

        # The model in the request and the response can be inconsistent
        # I.e request can specify gpt-4o-mini but the response from the server will be
        # gpt-4o-mini-2025-11-01. We need the model to be consistent so that inferences
        # will be grouped correctly on the Hiddenlayer side
        model_name = (
            logging_obj.model if logging_obj and logging_obj.model else "unknown"
        )
        hl_request_metadata = {"model": model_name}

        # We need the hiddenlayer project id and requester id on both the input and output
        # Since headers aren't available on the response back from the model, we get them
        # from the logging object. It ends up working out that on the request, we parse the
        # hiddenlayer params from the raw request and then retrieve those same headers
        # from the logger object on the response from the model.
        headers = request_data.get("proxy_server_request", {}).get("headers", {})
        if not headers and logging_obj and logging_obj.model_call_details:
            headers = (
                logging_obj.model_call_details.get("litellm_params", {})
                .get("metadata", {})
                .get("headers", {})
            )

        hl_request_metadata["requester_id"] = (
            headers.get("hl-requester-id") or "LiteLLM"
        )
        project_id = headers.get("hl-project-id")

        if scan_params := inputs.get("structured_messages"):
            last_msg = scan_params[-1]
            result = await self._call_hiddenlayer(
                project_id,
                hl_request_metadata,
                {"messages": [{"role": last_msg.get("role", "user"), "content": last_msg.get("content", "")}]},
                input_type,
            )
        elif text := inputs.get("texts"):
            result = await self._call_hiddenlayer(
                project_id,
                hl_request_metadata,
                {"messages": [{"role": "user", "content": text[-1]}]},
                input_type,
            )
        else:
            result = {}

        if result.get("evaluation", {}).get("action") == HiddenlayerAction.BLOCK:
            detected_reasons = [
                entry.get("name", "unknown")
                for entry in result.get("analysis", [])
                if entry.get("detected")
            ]
            threat_level = result.get("evaluation", {}).get("threat_level")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Violated guardrail policy",
                    "hiddenlayer_guardrail_response": HiddenlayerMessages.BLOCK_MESSAGE.value,
                    "block_reasons": detected_reasons,
                    "threat_level": threat_level,
                },
            )

        if result.get("evaluation", {}).get("action") == HiddenlayerAction.REDACT:
            modified_data = result.get("modified_data", {})
            if modified_data.get("input") and input_type == "request":
                inputs["texts"] = [modified_data["input"]["messages"][-1]["content"]]
                inputs["structured_messages"] = modified_data["input"]["messages"]

            if modified_data.get("output") and input_type == "response":
                inputs["texts"] = [modified_data["output"]["messages"][-1]["content"]]

        return inputs

    async def _call_hiddenlayer(
        self,
        project_id: str | None,
        metadata: dict[str, str],
        payload: dict[str, Any],
        input_type: Literal["request", "response"],
    ) -> dict[str, Any]:
        data: dict[str, Any] = {"metadata": metadata}

        if input_type == "request":
            data["input"] = payload
        else:
            data["output"] = payload

        headers = {
            "Content-Type": "application/json",
            "hl-runtime-edge-provider": "litellm",
            "hl-runtime-edge-provider-version": "1"
        }

        if project_id:
            headers["HL-Project-Id"] = project_id

        if self.jwt_token:
            headers["Authorization"] = f"Bearer {self.jwt_token}"

        try:
            response = await self._http_client.post(
                f"{self.api_base}/detection/v1/interactions",
                json=data,
                headers=headers,
            )
            response.raise_for_status()
            result = response.json()

            verbose_proxy_logger.debug(f"Hiddenlayer reponse: {result}")

            return result
        except HTTPStatusError as e:
            # Try the request again by refreshing the jwt if we get 401
            # since the Hiddenlayer jwt timeout is an hour and this is
            # a long lived session application
            if e.response.status_code == 401 and self.jwt_token is not None:
                verbose_proxy_logger.debug(
                    "Unable to authenticate to Hiddenlayer, JWT token is invalid or expired, trying to refresh the token."
                )
                self.jwt_token = self.refresh_jwt_func()
                headers["Authorization"] = f"Bearer {self.jwt_token}"
                response = await self._http_client.post(
                    f"{self.api_base}/detection/v1/interactions",
                    json=data,
                    headers=headers,
                )
            else:
                raise e

            response.raise_for_status()
            result = response.json()

            verbose_proxy_logger.debug(f"Hiddenlayer reponse: {result}")
            return result

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.hiddenlayer import (
            HiddenlayerGuardrailConfigModel,
        )

        return HiddenlayerGuardrailConfigModel

class HiddenlayerGuardrailV2(CustomGuardrail):
    """Custom guardrail wrapper for HiddenLayer's safety checks."""

    def __init__(
        self,
        api_id: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        auth_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.hiddenlayer_client_id = api_id or os.getenv("HIDDENLAYER_CLIENT_ID")
        self.hiddenlayer_client_secret = api_key or os.getenv(
            "HIDDENLAYER_CLIENT_SECRET"
        )
        self.api_base = (
            api_base
            or os.getenv("HIDDENLAYER_API_BASE")
            or "https://api.hiddenlayer.ai"
        )
        self.jwt_token = None

        auth_url = (
            auth_url
            or os.getenv("HIDDENLAYER_AUTH_URL")
            or "https://auth.hiddenlayer.ai"
        )

        if is_saas(self.api_base):
            if not self.hiddenlayer_client_id:
                raise RuntimeError(
                    "`api_id` cannot be None when using the SaaS version of HiddenLayer."
                )

            if not self.hiddenlayer_client_secret:
                raise RuntimeError(
                    "`api_key` cannot be None when using the SaaS version of HiddenLayer."
                )

            self.jwt_token = _get_jwt(
                auth_url=auth_url,
                api_id=self.hiddenlayer_client_id,
                api_key=self.hiddenlayer_client_secret,
            )
            self.refresh_jwt_func = lambda: _get_jwt(
                auth_url=auth_url,
                api_id=self.hiddenlayer_client_id,
                api_key=self.hiddenlayer_client_secret,
            )

        self._http_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )
        super().__init__(**kwargs)

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        """Validate (and optionally redact) text via HiddenLayer before/after LLM calls."""

        # We need the hiddenlayer project id and requester id on both the input and output
        # Since headers aren't available on the response back from the model, we get them
        # from the logging object. It ends up working out that on the request, we parse the
        # hiddenlayer params from the raw request and then retrieve those same headers
        # from the logger object on the response from the model.
        headers = request_data.get("proxy_server_request", {}).get("headers", {})
        if not headers and logging_obj and logging_obj.model_call_details:
            headers = (
                logging_obj.model_call_details.get("litellm_params", {})
                .get("metadata", {})
                .get("headers", {})
            )

        # put our roundtrip id in the header to the model so we get it on the way back from the model
        if "hl-roundtrip-id" not in headers:
            proxy_req = request_data.get("proxy_server_request")
            if proxy_req is not None and "headers" in proxy_req:
                proxy_req["headers"]["hl-roundtrip-id"] = str(uuid4())
                headers["hl-roundtrip-id"] = proxy_req["headers"]["hl-roundtrip-id"]

        hl_headers = {h.lower():v for h,v in headers.items() if h.lower().startswith("hl-")} 

        if "hl-requester-id" not in hl_headers:
            hl_headers["hl-requester-id"] = "LiteLLM"

        if input_type == "request":
            payload = {
                "messages": inputs.get("structured_messages"),
                "model": inputs.get("model"),
                "tools": inputs.get("tools")
            }
        else:
            if inputs.get("texts"):
                payload = {
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": inputs["texts"][0] if inputs.get("texts") else "",
                            },
                            "finish_reason": "stop",
                        }
                    ]
                }
            elif tool_calls := inputs.get("tool_calls"):
                payload = tool_calls
            else:
                payload = {}

        response = await self._call_hiddenlayer(
            payload,  # ty:ignore[invalid-argument-type]
            input_type,
            hl_headers
        )
        output = response.json()

        if response.headers.get("hl-runtime-action", "").lower() == "block":
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Violated guardrail policy",
                    "hiddenlayer_guardrail_response": HiddenlayerMessages.BLOCK_MESSAGE.value,
            })

        new_texts = []
        if input_type == "request":
            inputs["structured_messages"] = output 

            for message in output.get("messages", []):
                if content := message.get("content", ""):
                    new_texts.append(content)
                
            inputs["texts"] = new_texts
                
        elif input_type == "response" and inputs.get("texts"):
            inputs["texts"] = [output["choices"][-1]["message"]["content"]]
        elif input_type == "response" and inputs.get("tool_calls"):
            inputs["tool_calls"] = output

        return inputs

    async def _call_hiddenlayer(
        self,
        payload: dict[str, Any],
        input_type: Literal["request", "response"],
        hl_headers: dict[str, str]
    ) -> httpx.Response:

        if input_type == "request":
            path = "detection/v2/request-evaluations"
        else:
            path = "detection/v2/response-evaluations"

        headers = {
            "Content-Type": "application/json",
            "hl-runtime-edge-provider": "litellm",
            "hl-runtime-edge-provider-version": "2"
        }
        if self.jwt_token:
            headers["Authorization"] = f"Bearer {self.jwt_token}"

        headers.update(hl_headers)

        try:
            response = await self._http_client.post(
                f"{self.api_base}/{path}",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()

            verbose_proxy_logger.debug(f"Hiddenlayer reponse: {response}")

            return response
        except HTTPStatusError as e:
            # Try the request again by refreshing the jwt if we get 401
            # since the Hiddenlayer jwt timeout is an hour and this is
            # a long lived session application
            if e.response.status_code == 401 and self.jwt_token is not None:
                verbose_proxy_logger.debug(
                    "Unable to authenticate to Hiddenlayer, JWT token is invalid or expired, trying to refresh the token."
                )
                self.jwt_token = self.refresh_jwt_func()
                headers["Authorization"] = f"Bearer {self.jwt_token}"
                response = await self._http_client.post(
                    f"{self.api_base}/{path}",
                    json=payload,
                    headers=headers,
                )
            else:
                raise e

            response.raise_for_status()

            verbose_proxy_logger.debug(f"Hiddenlayer reponse: {response}")
            return response

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.hiddenlayer import (
            HiddenlayerGuardrailConfigModel,
        )

        return HiddenlayerGuardrailConfigModel
