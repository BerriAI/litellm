"""
e2b sandbox provider.

Talks to e2b's REST API directly over httpx (no e2b SDK dependency):
  - create: POST {api_base}/sandboxes
  - run:    POST https://{JUPYTER_PORT}-{sandboxID}.{domain}/execute  (NDJSON stream)
  - delete: DELETE {api_base}/sandboxes/{sandboxID}
"""

import json
from typing import List, Optional, Union, cast

import httpx

from litellm.llms.base_llm.sandbox.transformation import (
    BaseSandboxConfig,
    CodeExecutionResult,
    ContainerHandle,
)
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    get_async_httpx_client,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.custom_http import httpxSpecialProvider

E2B_API_BASE = "https://api.e2b.app"
E2B_DEFAULT_TEMPLATE = "code-interpreter-v1"
E2B_DEFAULT_DOMAIN = "e2b.app"
JUPYTER_PORT = 49999


class E2BSandboxConfig(BaseSandboxConfig):
    def _http(self, client: Optional[AsyncHTTPHandler]) -> AsyncHTTPHandler:
        if client is not None:
            return client
        return get_async_httpx_client(llm_provider=httpxSpecialProvider.Sandbox)

    def validate_environment(self, api_key: Optional[str] = None, **kwargs) -> str:
        key = api_key or get_secret_str("E2B_API_KEY")
        if not key:
            raise ValueError("E2B API key not set. Set E2B_API_KEY or pass api_key=...")
        return key

    async def acreate_sandbox(
        self,
        *,
        template: Optional[str] = None,
        timeout: Optional[int] = None,
        allow_internet_access: bool = True,
        api_key: Optional[str] = None,
        metadata: Optional[dict] = None,
        client: Optional[AsyncHTTPHandler] = None,
        **kwargs,
    ) -> ContainerHandle:
        key = self.validate_environment(api_key=api_key)
        body = {
            "templateID": template or E2B_DEFAULT_TEMPLATE,
            "timeout": timeout or 300,
            "secure": True,
            "allow_internet_access": allow_internet_access,
        }
        if metadata:
            body["metadata"] = metadata

        response = cast(
            httpx.Response,
            await self._http(client).post(
                url=f"{E2B_API_BASE}/sandboxes",
                headers={"X-API-Key": key, "Content-Type": "application/json"},
                json=body,
            ),
        )
        data = response.json()

        handle = ContainerHandle(
            id=data["sandboxID"],
            provider="e2b",
            domain=data.get("domain") or E2B_DEFAULT_DOMAIN,
        )
        handle._hidden_params = {
            "envd_access_token": data.get("envdAccessToken"),
            "traffic_access_token": data.get("trafficAccessToken"),
            "api_key": key,
        }
        return handle

    async def arun_code(
        self,
        *,
        container: Union[ContainerHandle, str],
        code: str,
        api_key: Optional[str] = None,
        env_vars: Optional[dict] = None,
        client: Optional[AsyncHTTPHandler] = None,
        **kwargs,
    ) -> CodeExecutionResult:
        handle = self._as_handle(container)

        headers = {"Content-Type": "application/json"}
        token = handle._hidden_params.get("envd_access_token")
        if token:
            headers["X-Access-Token"] = token
        traffic_token = handle._hidden_params.get("traffic_access_token")
        if traffic_token:
            headers["E2B-Traffic-Access-Token"] = traffic_token

        url = f"https://{JUPYTER_PORT}-{handle.id}.{handle.domain}/execute"
        response = cast(
            httpx.Response,
            await self._http(client).post(
                url=url,
                headers=headers,
                json={"code": code, "context_id": None, "env_vars": env_vars},
                stream=True,
            ),
        )
        lines = [line async for line in response.aiter_lines()]
        return self._parse_lines(lines)

    async def adelete_sandbox(
        self,
        *,
        container: Union[ContainerHandle, str],
        api_key: Optional[str] = None,
        client: Optional[AsyncHTTPHandler] = None,
        **kwargs,
    ) -> bool:
        handle = self._as_handle(container)
        key = (
            api_key
            or handle._hidden_params.get("api_key")
            or self.validate_environment()
        )
        response = cast(
            httpx.Response,
            await self._http(client).delete(
                url=f"{E2B_API_BASE}/sandboxes/{handle.id}",
                headers={"X-API-Key": key},
            ),
        )
        return 200 <= response.status_code < 300

    @staticmethod
    def _as_handle(container: Union[ContainerHandle, str]) -> ContainerHandle:
        if isinstance(container, ContainerHandle):
            return container
        handle = ContainerHandle(
            id=str(container), provider="e2b", domain=E2B_DEFAULT_DOMAIN
        )
        handle._hidden_params = {}
        return handle

    @staticmethod
    def _parse_lines(lines: List[str]) -> CodeExecutionResult:
        messages = tuple(
            json.loads(stripped)
            for stripped in (line.strip() for line in lines)
            if stripped
        )

        def of_type(message_type: str):
            return (m for m in messages if m.get("type") == message_type)

        error = next(
            (
                {key: m.get(key) for key in ("name", "value", "traceback")}
                for m in of_type("error")
            ),
            None,
        )
        execution_count = next(
            (m.get("execution_count") for m in of_type("number_of_executions")),
            None,
        )

        return CodeExecutionResult(
            stdout="".join(m.get("text", "") for m in of_type("stdout")),
            stderr="".join(m.get("text", "") for m in of_type("stderr")),
            results=[
                {k: v for k, v in m.items() if k != "type"}
                for m in of_type("result")
            ],
            error=error,
            execution_count=execution_count,
        )
