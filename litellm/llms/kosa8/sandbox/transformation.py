"""
kosa8 sandbox provider.

Talks to the kosa8 Cloud REST API directly over httpx (no SDK dependency):
  - create: POST {api_base}/v1/sandboxes
  - run:    POST {api_base}/v1/sandboxes/{id}/code  (stateful Python kernel)
  - delete: DELETE {api_base}/v1/sandboxes/{id}
"""

from collections.abc import Mapping
from typing import Final, cast

import httpx
from pydantic import ConfigDict, TypeAdapter

from litellm.litellm_core_utils.hidden_params import HIDDEN_PARAMS_ATTR
from litellm.litellm_core_utils.url_utils import encode_url_path_segment, join_api_base
from litellm.llms.base_llm.sandbox.transformation import (
    SANDBOX_MAX_OUTPUT_BYTES,
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

KOSA8_API_BASE: Final = "https://api.kosa8.com"
KOSA8_DEFAULT_TEMPLATE: Final = "python"
DEFAULT_SANDBOX_TIMEOUT: Final = 300
MIN_SANDBOX_TIMEOUT: Final = 60

_HIDDEN: Final = ConfigDict(hide_input_in_errors=True)
_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object], config=_HIDDEN)
_TEXT: Final = TypeAdapter(str, config=ConfigDict(strict=True, hide_input_in_errors=True))
_RESULTS: Final = TypeAdapter(list[dict[str, object]], config=_HIDDEN)
_ERROR: Final = TypeAdapter(dict[str, object] | None, config=_HIDDEN)
_COUNT: Final = TypeAdapter(int | None, config=ConfigDict(strict=True, hide_input_in_errors=True))


class Kosa8SandboxConfig(BaseSandboxConfig):
    def _http(self, client: AsyncHTTPHandler | None) -> AsyncHTTPHandler:
        if client is not None:
            return client
        return get_async_httpx_client(llm_provider=httpxSpecialProvider.Sandbox)

    def validate_environment(self, api_key: str | None = None, **kwargs: object) -> str:
        key: Final = api_key or get_secret_str("KOSA8_API_KEY")
        if not key:
            raise ValueError("kosa8 API key not set. Set KOSA8_API_KEY or pass api_key=...")
        return key

    @staticmethod
    def _api_base(api_base: object) -> str:
        return str(api_base or get_secret_str("KOSA8_API_BASE") or KOSA8_API_BASE)

    async def acreate_sandbox(
        self,
        *,
        template: str | None = None,
        timeout: int | None = None,
        allow_internet_access: bool | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        client: AsyncHTTPHandler | None = None,
        **kwargs: object,
    ) -> ContainerHandle:
        key: Final = self.validate_environment(api_key=api_key)
        base: Final = self._api_base(api_base)
        lifetime: Final = DEFAULT_SANDBOX_TIMEOUT if timeout is None else max(timeout, MIN_SANDBOX_TIMEOUT)
        egress: Final = {} if allow_internet_access is None else {"egress": {"internet": allow_internet_access}}

        response: Final = await self._http(client).post(
            url=join_api_base(base, "v1/sandboxes"),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "template": template or KOSA8_DEFAULT_TEMPLATE,
                "ttl_s": lifetime,
                "persistent": True,
                **egress,
            },
        )
        sandbox: Final = _JSON_OBJECT.validate_python(_JSON_OBJECT.validate_python(response.json())["sandbox"])

        handle: Final = ContainerHandle.model_validate({"id": sandbox["id"], "provider": "kosa8"})
        handle.hidden_params = {"api_key": key, "api_base": base}
        return handle

    async def arun_code(
        self,
        *,
        container: ContainerHandle | str,
        code: str,
        api_key: str | None = None,
        api_base: str | None = None,
        client: AsyncHTTPHandler | None = None,
        **kwargs: object,
    ) -> CodeExecutionResult:
        handle: Final = self._as_handle(container)
        key, base = self._credentials(handle, api_key, api_base)
        response: Final = await self._http(client).post(
            url=join_api_base(base, f"v1/sandboxes/{encode_url_path_segment(handle.id)}/code"),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"code": code, "language": "python"},
        )
        if len(response.content) > SANDBOX_MAX_OUTPUT_BYTES:
            raise ValueError(
                f"Sandbox output exceeded {SANDBOX_MAX_OUTPUT_BYTES} bytes; aborting to avoid unbounded memory use."
            )
        data: Final = _JSON_OBJECT.validate_python(response.json())
        return CodeExecutionResult(
            stdout=_TEXT.validate_python(data.get("stdout") or ""),
            stderr=_TEXT.validate_python(data.get("stderr") or ""),
            results=_RESULTS.validate_python(data.get("results") or []),
            error=_ERROR.validate_python(data.get("error")),
            execution_count=_COUNT.validate_python(data.get("execution_count")),
        )

    async def adelete_sandbox(
        self,
        *,
        container: ContainerHandle | str,
        api_key: str | None = None,
        api_base: str | None = None,
        client: AsyncHTTPHandler | None = None,
        **kwargs: object,
    ) -> bool:
        handle: Final = self._as_handle(container)
        key, base = self._credentials(handle, api_key, api_base)
        try:
            response: Final = await self._http(client).delete(
                url=join_api_base(base, f"v1/sandboxes/{encode_url_path_segment(handle.id)}"),
                headers={"Authorization": f"Bearer {key}"},
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return False
            raise
        return 200 <= response.status_code < 300

    def _credentials(self, handle: ContainerHandle, api_key: str | None, api_base: str | None) -> tuple[str, str]:
        hidden_params: Final = cast(  # cast-ok: preserve mapping operations on the validated handle
            dict[str, object], getattr(handle, HIDDEN_PARAMS_ATTR)
        )
        stored_key: Final = hidden_params.get("api_key")
        key: Final = api_key or (stored_key if isinstance(stored_key, str) else None) or self.validate_environment()
        return key, self._api_base(api_base or hidden_params.get("api_base"))

    @staticmethod
    def _as_handle(container: ContainerHandle | str) -> ContainerHandle:
        if isinstance(container, ContainerHandle):
            return container
        handle: Final = ContainerHandle(id=str(container), provider="kosa8")
        handle.hidden_params = {}
        return handle
