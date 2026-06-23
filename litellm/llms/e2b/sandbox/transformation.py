"""
e2b sandbox provider.

Talks to e2b's REST API directly over httpx (no e2b SDK dependency):
  - create: POST {api_base}/sandboxes
  - run:    POST https://{JUPYTER_PORT}-{sandboxID}.{domain}/execute  (NDJSON stream)
  - delete: DELETE {api_base}/sandboxes/{sandboxID}
"""

import json
from typing import Union, cast

import httpx

from litellm.llms.base_llm.sandbox.transformation import (
    BaseSandboxConfig,
    CodeExecutionResult,
    ContainerHandle,
    SANDBOX_MAX_OUTPUT_BYTES,
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
DEFAULT_SANDBOX_TIMEOUT = 300
MAX_OUTPUT_BYTES = SANDBOX_MAX_OUTPUT_BYTES


class E2BSandboxConfig(BaseSandboxConfig):
    def _http(self, client: AsyncHTTPHandler | None) -> AsyncHTTPHandler:
        if client is not None:
            return client
        return get_async_httpx_client(llm_provider=httpxSpecialProvider.Sandbox)

    def validate_environment(self, api_key: str | None = None, **kwargs) -> str:
        key = api_key or get_secret_str("E2B_API_KEY")
        if not key:
            raise ValueError("E2B API key not set. Set E2B_API_KEY or pass api_key=...")
        return key

    async def acreate_sandbox(
        self,
        *,
        template: str | None = None,
        timeout: int | None = None,
        allow_internet_access: bool | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        metadata: dict | None = None,
        client: AsyncHTTPHandler | None = None,
        **kwargs,
    ) -> ContainerHandle:
        key = self.validate_environment(api_key=api_key)
        base = api_base or E2B_API_BASE
        body = {
            "templateID": template or E2B_DEFAULT_TEMPLATE,
            "timeout": timeout if timeout is not None else DEFAULT_SANDBOX_TIMEOUT,
            "secure": True,
            "allow_internet_access": (
                True if allow_internet_access is None else allow_internet_access
            ),
        }
        if metadata:
            body["metadata"] = metadata

        response = cast(
            httpx.Response,
            await self._http(client).post(
                url=f"{base}/sandboxes",
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
            "api_base": base,
        }
        return handle

    async def arun_code(
        self,
        *,
        container: Union[ContainerHandle, str],
        code: str,
        api_key: str | None = None,
        env_vars: dict | None = None,
        client: AsyncHTTPHandler | None = None,
        **kwargs,
    ) -> CodeExecutionResult:
        handle = self._as_handle(container)

        token = handle._hidden_params.get("envd_access_token")
        if not token:
            raise ValueError(
                "Cannot run code from a sandbox id alone. e2b secure sandboxes "
                "require the access token returned by acreate_sandbox; pass the "
                "ContainerHandle it returned instead of a bare sandbox id."
            )

        headers = {"Content-Type": "application/json", "X-Access-Token": token}
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
        lines = await self._read_capped_lines(response)
        return self._parse_lines(lines)

    async def adelete_sandbox(
        self,
        *,
        container: Union[ContainerHandle, str],
        api_key: str | None = None,
        api_base: str | None = None,
        client: AsyncHTTPHandler | None = None,
        **kwargs,
    ) -> bool:
        handle = self._as_handle(container)
        key = (
            api_key
            or handle._hidden_params.get("api_key")
            or self.validate_environment()
        )
        base = api_base or handle._hidden_params.get("api_base") or E2B_API_BASE
        try:
            response = cast(
                httpx.Response,
                await self._http(client).delete(
                    url=f"{base}/sandboxes/{handle.id}",
                    headers={"X-API-Key": key},
                ),
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return False
            raise
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
    def _parse_lines(lines: list[str]) -> CodeExecutionResult:
        def _try_parse(stripped: str):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return None

        messages = tuple(
            parsed
            for line in lines
            if (stripped := line.strip())
            if (parsed := _try_parse(stripped)) is not None
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
                {k: v for k, v in m.items() if k != "type"} for m in of_type("result")
            ],
            error=error,
            execution_count=execution_count,
        )
