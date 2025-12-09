from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from httpx import HTTPStatusError

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from urllib.parse import urlparse
import requests
from requests.auth import HTTPBasicAuth

from fastapi import HTTPException

from litellm.types.proxy.guardrails.guardrail_hooks.hiddenlayer import HiddenlayerAction, HiddenlayerMessages


def is_saas(host: str) -> bool:
    """Checks whether the connection is to the SaaS platform"""

    o = urlparse(host)

    if o.hostname and o.hostname.endswith("hiddenlayer.ai"):
        return True

    return False


def _get_jwt(api_id, api_key):
    auth_url = os.getenv("HL_AUTH_URL", "https://auth.hiddenlayer.ai")

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


class HiddenLayerGuardrail(CustomGuardrail):
    """Custom guardrail wrapper for HiddenLayer's safety checks."""

    def __init__(
        self,
        api_id: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.hiddenlayer_client_id = api_id or os.getenv("HIDDENLAYER_CLIENT_ID")
        self.hiddenlayer_client_secret = api_key or os.getenv("HIDDENLAYER_CLIENT_SECRET")
        self.api_base = api_base or os.getenv("HIDDENLAYER_API_BASE") or "https://api.hiddenlayer.ai"
        self.jwt_token = None

        if is_saas(self.api_base):
            if not self.hiddenlayer_client_id:
                raise EnvironmentError("`api_id` cannot be None when using the SaaS version of HiddenLayer.")

            if not self.hiddenlayer_client_secret:
                raise EnvironmentError("`api_key` cannot be None when using the SaaS version of HiddenLayer.")

            self.jwt_token = _get_jwt(api_id=self.hiddenlayer_client_id, api_key=self.hiddenlayer_client_secret)
            self.refresh_jwt_func = lambda: _get_jwt(
                api_id=self.hiddenlayer_client_id, api_key=self.hiddenlayer_client_secret
            )

        self._http_client = get_async_httpx_client(llm_provider=httpxSpecialProvider.GuardrailCallback)
        super().__init__(**kwargs)

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> str:
        """Validate (and optionally redact) text via HiddenLayer before/after LLM calls."""

        hl_request_metadata = {"model": logging_obj.model}

        # Try to get headers from request, if we can't - try getting them from the model request logger
        # Generally the output calls will get the headers from the model request logger
        headers = request_data.get("proxy_server_request", {}).get("headers", {})

        if not headers:
            headers = logging_obj.model_call_details.get("litellm_params", {}).get("metadata", {}).get("headers", {})

        hl_request_metadata["requester_id"] = headers.get("hl-requester-id") or "LiteLLM"

        project_id = headers.get("hl-project-id")

        if scan_params := inputs.get("structured_messages"):
            result = await self._call_hiddenlayer(
                project_id, hl_request_metadata, {"messages": scan_params}, input_type
            )
        elif text := inputs.get("texts"):
            result = await self._call_hiddenlayer(
                project_id, hl_request_metadata, {"messages": [{"role": "user", "content": text[-1]}]}, input_type
            )
        else:
            result = {}

        if result.get("evaluation", {}).get("action") == HiddenlayerAction.BLOCK:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Violated guardrail policy",
                    "hiddenlayer_guardrail_response": HiddenlayerMessages.BLOCK_MESSAGE,
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
        payload: dict[Literal["messages"], list[dict[str, str]]],
        input_type: Literal["request", "response"],
    ) -> dict:
        data = {"metadata": metadata}

        if input_type == "request":
            data["input"] = payload
        else:
            data["output"] = payload

        headers = {
            "Content-Type": "application/json",
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
        except HTTPStatusError as e:
            # Try the request again by refreshing the jwt if we get 401
            # since the Hiddenlayer jwt timeout is an hour and this is
            # a long lived session application
            if e.response.status_code == 401 and self.jwt_token is not None:
                self.jwt_token = self.refresh_jwt_func()
                headers["Authorization"] = f"Bearer {self.jwt_token}"
                response = await self._http_client.post(
                    f"{self.api_base}/detection/v1/interactions",
                    json=data,
                    headers=headers,
                )

        response.raise_for_status()
        result = response.json()

        return result
