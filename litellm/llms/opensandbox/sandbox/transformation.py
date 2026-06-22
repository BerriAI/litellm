import asyncio
import json
import time
from typing import Union, cast

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

OPEN_SANDBOX_API_BASE = "http://localhost:8080/v1"
OPEN_SANDBOX_DEFAULT_TEMPLATE = "opensandbox/code-interpreter:v1.1.0"
OPEN_SANDBOX_DEFAULT_ENTRYPOINT = ("/opt/code-interpreter/code-interpreter.sh",)
OPEN_SANDBOX_DEFAULT_LANGUAGE = "python"
OPEN_SANDBOX_DEFAULT_RESOURCE_LIMITS = {"cpu": "1", "memory": "2Gi"}
OPEN_SANDBOX_EXECD_PORT = 44772
DEFAULT_SANDBOX_TIMEOUT = 300
DEFAULT_READY_TIMEOUT = 30.0
DEFAULT_POLL_INTERVAL = 0.2
MAX_OUTPUT_BYTES = 10 * 1024 * 1024


class OpenSandboxSandboxConfig(BaseSandboxConfig):
    def _http(self, client: AsyncHTTPHandler | None) -> AsyncHTTPHandler:
        if client is not None:
            return client
        return get_async_httpx_client(llm_provider=httpxSpecialProvider.Sandbox)

    def validate_environment(self, api_key: str | None = None, **kwargs) -> str:
        if api_key is not None:
            return api_key
        return get_secret_str("OPEN_SANDBOX_API_KEY") or ""

    async def acreate_sandbox(
        self,
        *,
        template: str | None = None,
        timeout: int | None = None,
        allow_internet_access: bool = True,
        api_key: str | None = None,
        api_base: str | None = None,
        metadata: dict[str, str] | None = None,
        env_vars: dict[str, str] | None = None,
        resource_limits: dict[str, str] | None = None,
        resource_requests: dict[str, str] | None = None,
        entrypoint: list[str] | tuple[str, ...] | None = None,
        network_policy: dict[str, object] | None = None,
        secure_access: bool = False,
        use_server_proxy: bool = False,
        ready_timeout: float | int | None = None,
        poll_interval: float | int | None = None,
        client: AsyncHTTPHandler | None = None,
        **kwargs,
    ) -> ContainerHandle:
        key = self.validate_environment(api_key=api_key)
        base = self._api_base(api_base)
        body = self._create_body(
            template=template,
            timeout=timeout,
            allow_internet_access=allow_internet_access,
            metadata=metadata,
            env_vars=env_vars,
            resource_limits=resource_limits,
            resource_requests=resource_requests,
            entrypoint=entrypoint,
            network_policy=network_policy,
            secure_access=secure_access,
        )

        response = cast(
            httpx.Response,
            await self._http(client).post(
                url=f"{base}/sandboxes",
                headers=self._lifecycle_headers(key),
                json=body,
            ),
        )
        data = response.json()
        sandbox_id = str(data["id"])

        if self._sandbox_state(data) != "Running":
            await self._wait_until_running(
                sandbox_id=sandbox_id,
                api_base=base,
                headers=self._lifecycle_headers(key),
                client=client,
                ready_timeout=(
                    float(ready_timeout)
                    if ready_timeout is not None
                    else DEFAULT_READY_TIMEOUT
                ),
                poll_interval=(
                    float(poll_interval)
                    if poll_interval is not None
                    else DEFAULT_POLL_INTERVAL
                ),
            )

        endpoint, endpoint_headers = await self._get_execd_endpoint(
            sandbox_id=sandbox_id,
            api_base=base,
            headers=self._lifecycle_headers(key),
            use_server_proxy=use_server_proxy,
            client=client,
        )

        handle = ContainerHandle(id=sandbox_id, provider="opensandbox", domain=base)
        handle._hidden_params = {
            "api_base": base,
            "api_key": key,
            "execd_endpoint": endpoint,
            "execd_headers": endpoint_headers,
            "use_server_proxy": use_server_proxy,
        }
        return handle

    async def arun_code(
        self,
        *,
        container: Union[ContainerHandle, str],
        code: str,
        api_key: str | None = None,
        api_base: str | None = None,
        language: str = OPEN_SANDBOX_DEFAULT_LANGUAGE,
        use_server_proxy: bool = False,
        client: AsyncHTTPHandler | None = None,
        **kwargs,
    ) -> CodeExecutionResult:
        handle = await self._ensure_handle(
            container=container,
            api_key=api_key,
            api_base=api_base,
            use_server_proxy=use_server_proxy,
            client=client,
        )
        endpoint = str(handle._hidden_params["execd_endpoint"])
        endpoint_headers = self._as_str_dict(handle._hidden_params.get("execd_headers"))
        base = str(
            handle._hidden_params.get("api_base")
            or handle.domain
            or self._api_base(api_base)
        )
        lines = await self._post_code(
            url=f"{self._endpoint_base_url(endpoint, base)}/code",
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "Cache-Control": "no-cache",
                **endpoint_headers,
            },
            body={
                "code": code,
                "context": {"language": language},
            },
            client=client,
        )
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
        handle = self._as_handle(container, api_base=api_base)
        base = str(handle._hidden_params.get("api_base") or self._api_base(api_base))
        key = self._api_key(api_key=api_key, handle=handle)
        try:
            response = cast(
                httpx.Response,
                await self._http(client).delete(
                    url=f"{base}/sandboxes/{handle.id}",
                    headers=self._lifecycle_headers(key),
                ),
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return False
            raise
        return 200 <= response.status_code < 300

    async def _ensure_handle(
        self,
        *,
        container: Union[ContainerHandle, str],
        api_key: str | None,
        api_base: str | None,
        use_server_proxy: bool,
        client: AsyncHTTPHandler | None,
    ) -> ContainerHandle:
        handle = self._as_handle(container, api_base=api_base)
        if handle._hidden_params.get("execd_endpoint"):
            return handle

        base = str(handle._hidden_params.get("api_base") or self._api_base(api_base))
        key = self._api_key(api_key=api_key, handle=handle)
        resolved_use_server_proxy = bool(
            handle._hidden_params.get("use_server_proxy", use_server_proxy)
        )
        endpoint, endpoint_headers = await self._get_execd_endpoint(
            sandbox_id=handle.id,
            api_base=base,
            headers=self._lifecycle_headers(key),
            use_server_proxy=resolved_use_server_proxy,
            client=client,
        )
        handle.domain = base
        handle._hidden_params = {
            **handle._hidden_params,
            "api_base": base,
            "api_key": key,
            "execd_endpoint": endpoint,
            "execd_headers": endpoint_headers,
            "use_server_proxy": resolved_use_server_proxy,
        }
        return handle

    async def _wait_until_running(
        self,
        *,
        sandbox_id: str,
        api_base: str,
        headers: dict[str, str],
        client: AsyncHTTPHandler | None,
        ready_timeout: float,
        poll_interval: float,
    ) -> None:
        deadline = time.monotonic() + ready_timeout
        while True:
            response = cast(
                httpx.Response,
                await self._http(client).get(
                    url=f"{api_base}/sandboxes/{sandbox_id}",
                    headers=headers,
                ),
            )
            self._raise_for_status(response)
            data = response.json()
            state = self._sandbox_state(data)
            if state == "Running":
                return
            if state in {"Failed", "Stopping", "Terminated"}:
                raise ValueError(f"OpenSandbox sandbox {sandbox_id} entered {state}")
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"OpenSandbox sandbox {sandbox_id} was not Running within "
                    f"{ready_timeout} seconds"
                )
            await asyncio.sleep(poll_interval)

    async def _get_execd_endpoint(
        self,
        *,
        sandbox_id: str,
        api_base: str,
        headers: dict[str, str],
        use_server_proxy: bool,
        client: AsyncHTTPHandler | None,
    ) -> tuple[str, dict[str, str]]:
        response = cast(
            httpx.Response,
            await self._http(client).get(
                url=f"{api_base}/sandboxes/{sandbox_id}/endpoints/{OPEN_SANDBOX_EXECD_PORT}",
                headers=headers,
                params={"use_server_proxy": use_server_proxy},
            ),
        )
        self._raise_for_status(response)
        data = response.json()
        endpoint = data.get("endpoint")
        if not endpoint:
            raise ValueError(
                f"OpenSandbox did not return an execd endpoint for {sandbox_id}"
            )
        return str(endpoint), self._as_str_dict(data.get("headers"))

    async def _post_code(
        self,
        *,
        url: str,
        headers: dict[str, str],
        body: dict[str, object],
        client: AsyncHTTPHandler | None,
    ) -> list[str]:
        timeout = httpx.Timeout(connect=30.0, read=None, write=30.0, pool=None)
        response = cast(
            httpx.Response,
            await self._http(client).post(
                url=url,
                headers=headers,
                timeout=timeout,
                json=body,
                stream=True,
            ),
        )
        return await self._read_capped_lines(response)

    def _api_key(self, *, api_key: str | None, handle: ContainerHandle) -> str:
        if api_key is not None:
            return api_key
        if "api_key" in handle._hidden_params:
            return str(handle._hidden_params["api_key"])
        return self.validate_environment()

    @staticmethod
    def _create_body(
        *,
        template: str | None,
        timeout: int | None,
        allow_internet_access: bool,
        metadata: dict[str, str] | None,
        env_vars: dict[str, str] | None,
        resource_limits: dict[str, str] | None,
        resource_requests: dict[str, str] | None,
        entrypoint: list[str] | tuple[str, ...] | None,
        network_policy: dict[str, object] | None,
        secure_access: bool,
    ) -> dict[str, object]:
        body: dict[str, object] = {
            "image": {"uri": template or OPEN_SANDBOX_DEFAULT_TEMPLATE},
            "entrypoint": list(entrypoint or OPEN_SANDBOX_DEFAULT_ENTRYPOINT),
            "timeout": timeout if timeout is not None else DEFAULT_SANDBOX_TIMEOUT,
            "resourceLimits": resource_limits or OPEN_SANDBOX_DEFAULT_RESOURCE_LIMITS,
        }
        if metadata:
            body["metadata"] = metadata
        if env_vars:
            body["env"] = env_vars
        if resource_requests:
            body["resourceRequests"] = resource_requests
        if network_policy is not None:
            body["networkPolicy"] = network_policy
        elif not allow_internet_access:
            body["networkPolicy"] = {"defaultAction": "deny", "egress": []}
        if secure_access:
            body["secureAccess"] = True
        return body

    @staticmethod
    def _sandbox_state(data: object) -> str | None:
        if not isinstance(data, dict):
            return None
        status = data.get("status")
        if not isinstance(status, dict):
            return None
        state = status.get("state")
        return str(state) if state is not None else None

    @staticmethod
    def _as_str_dict(value: object) -> dict[str, str]:
        if not isinstance(value, dict):
            return {}
        return {str(k): str(v) for k, v in value.items()}

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        response.raise_for_status()

    @staticmethod
    def _api_base(api_base: str | None) -> str:
        return (
            api_base or get_secret_str("OPEN_SANDBOX_API_BASE") or OPEN_SANDBOX_API_BASE
        ).rstrip("/")

    @staticmethod
    def _lifecycle_headers(api_key: str) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["OPEN-SANDBOX-API-KEY"] = api_key
        return headers

    @staticmethod
    def _endpoint_base_url(endpoint: str, api_base: str) -> str:
        normalized_endpoint = endpoint.rstrip("/")
        if normalized_endpoint.startswith(("http://", "https://")):
            return normalized_endpoint
        protocol = api_base.split("://", 1)[0] if "://" in api_base else "http"
        return f"{protocol}://{normalized_endpoint}"

    @staticmethod
    def _as_handle(
        container: Union[ContainerHandle, str], *, api_base: str | None
    ) -> ContainerHandle:
        if isinstance(container, ContainerHandle):
            return container
        handle = ContainerHandle(
            id=str(container),
            provider="opensandbox",
            domain=OpenSandboxSandboxConfig._api_base(api_base),
        )
        handle._hidden_params = {}
        return handle

    @staticmethod
    async def _read_capped_lines(response: httpx.Response) -> list[str]:
        lines: list[str] = []
        total = 0
        async for line in response.aiter_lines():
            total += len(line.encode("utf-8"))
            if total > MAX_OUTPUT_BYTES:
                raise ValueError(
                    f"Sandbox output exceeded {MAX_OUTPUT_BYTES} bytes; aborting to "
                    "avoid unbounded memory use."
                )
            lines.append(line)
        return lines

    @staticmethod
    def _parse_lines(lines: list[str]) -> CodeExecutionResult:
        messages = tuple(
            event
            for line in lines
            for event in (OpenSandboxSandboxConfig._parse_sse_line(line),)
            if event is not None
        )

        def of_type(message_type: str):
            return (m for m in messages if m.get("type") == message_type)

        error = next(
            (OpenSandboxSandboxConfig._normalize_error(m) for m in of_type("error")),
            None,
        )
        execution_count = next(
            (
                OpenSandboxSandboxConfig._as_int(m.get("execution_count"))
                for m in of_type("execution_count")
                if OpenSandboxSandboxConfig._as_int(m.get("execution_count"))
                is not None
            ),
            None,
        )

        return CodeExecutionResult(
            stdout="".join(str(m.get("text", "")) for m in of_type("stdout")),
            stderr="".join(str(m.get("text", "")) for m in of_type("stderr")),
            results=[
                OpenSandboxSandboxConfig._normalize_result(m) for m in of_type("result")
            ],
            error=error,
            execution_count=execution_count,
        )

    @staticmethod
    def _parse_sse_line(line: str) -> dict[str, object] | None:
        stripped = line.strip()
        if not stripped or stripped.startswith(
            (
                ":",
                "event:",
                "id:",
                "retry:",
            )
        ):
            return None
        data = stripped[5:].strip() if stripped.startswith("data:") else stripped
        if not data:
            return None
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        if "type" not in parsed and "code" in parsed and "message" in parsed:
            return {
                "type": "error",
                "error": {
                    "ename": str(parsed["code"]),
                    "evalue": str(parsed["message"]),
                    "traceback": [],
                },
            }
        return parsed

    @staticmethod
    def _normalize_result(message: dict[str, object]) -> dict[str, object]:
        results = message.get("results")
        if isinstance(results, dict):
            return {str(k): v for k, v in results.items()}
        return {
            str(k): v
            for k, v in message.items()
            if k not in {"type", "timestamp", "execution_count"}
        }

    @staticmethod
    def _normalize_error(message: dict[str, object]) -> dict[str, object]:
        raw_error = message.get("error")
        if isinstance(raw_error, dict):
            name = raw_error.get("ename") or raw_error.get("name") or ""
            value = raw_error.get("evalue") or raw_error.get("value") or ""
            traceback = raw_error.get("traceback") or []
            return {
                "name": name,
                "value": value,
                "traceback": traceback,
            }
        return {
            "name": message.get("name") or "",
            "value": message.get("value") or message.get("text") or "",
            "traceback": message.get("traceback") or [],
        }

    @staticmethod
    def _as_int(value: object) -> int | None:
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return None
        return None
