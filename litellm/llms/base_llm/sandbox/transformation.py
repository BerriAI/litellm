"""
Base Sandbox transformation configuration.

A sandbox provider runs an executable string inside an isolated container and
returns whatever the sandbox produced. The lifecycle is create container ->
run code -> delete container; `code_interpreter_tool` combines all three.
"""

import builtins
import shlex
import time
from collections.abc import Mapping
from functools import cache
from importlib.resources import files
from typing import Final, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, TypeAdapter

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, get_async_httpx_client
from litellm.llms.custom_httpx.sse_lines import aiter_sse_lines
from litellm.types.llms.base import LiteLLMPydanticObjectBase
from litellm.types.llms.custom_http import httpxSpecialProvider

SANDBOX_MAX_OUTPUT_BYTES: Final = 10 * 1024 * 1024


class SandboxCost(LiteLLMPydanticObjectBase):
    """What a sandbox has cost so far. `cost_usd` is None when it cannot be known, never a guessed 0."""

    model_config = ConfigDict(frozen=True)

    cost_usd: float | None = None
    source: Literal["provider", "estimate", "unknown"] = "unknown"
    seconds: float | None = None


class SandboxPrice(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    cost_per_vcpu_second: float = 0.0
    cost_per_gib_second: float = 0.0
    cost_per_disk_gib_second: float = 0.0


_SANDBOX_PRICES: Final = TypeAdapter(dict[str, SandboxPrice])


@cache
def get_sandbox_prices() -> Mapping[str, SandboxPrice]:
    return _SANDBOX_PRICES.validate_json(files("litellm").joinpath("sandbox_prices.json").read_bytes())


class ContainerHandle(LiteLLMPydanticObjectBase):
    """A live sandbox container. Carries everything needed to reach it again."""

    id: str
    provider: str
    domain: str | None = None
    created_at: float | None = None
    ended_at: float | None = None
    expires_at: float | None = None
    vcpus: float | None = None
    memory_gib: float | None = None
    disk_gib: float | None = None
    final_cost: SandboxCost | None = None

    model_config = {"extra": "allow"}

    _hidden_params: dict[str, builtins.object] = PrivateAttr(default_factory=dict)

    @property
    def hidden_params(self) -> dict[str, builtins.object]:  # mutable-ok: API requires mutation
        return self._hidden_params

    @hidden_params.setter
    def hidden_params(self, hidden_params: dict[str, builtins.object]) -> None:  # mutable-ok: API requires mutation
        self._hidden_params = hidden_params


def estimate_sandbox_cost(container: ContainerHandle | str, now: float | None = None) -> SandboxCost:
    """Price the sandbox's allocated resources over its lifetime from `sandbox_prices.json`."""
    if not isinstance(container, ContainerHandle) or container.created_at is None:
        return SandboxCost()
    current: Final = time.time() if now is None else now
    running_until: Final = current if container.expires_at is None else min(current, container.expires_at)
    end: Final = container.ended_at if container.ended_at is not None else running_until
    seconds: Final = max(end - container.created_at, 0.0)
    price: Final = get_sandbox_prices().get(container.provider)
    if price is None or container.memory_gib is None or container.vcpus is None:
        return SandboxCost(seconds=seconds)
    per_second: Final = (
        price.cost_per_vcpu_second * container.vcpus
        + price.cost_per_gib_second * container.memory_gib
        + price.cost_per_disk_gib_second * (container.disk_gib or 0.0)
    )
    return SandboxCost(cost_usd=per_second * seconds, source="estimate", seconds=seconds)


def shell_command(code: str, language: str, max_output_bytes: int = SANDBOX_MAX_OUTPUT_BYTES) -> str:
    """
    Wraps the code so the sandbox itself stops each output stream one byte past the cap,
    which ends the command instead of letting it stream without limit into our memory.
    """
    if language not in ("bash", "python"):
        raise ValueError(f"Unsupported sandbox language: {language}. Use 'python' or 'bash'.")
    command: Final = code if language == "bash" else f"python3 -c {shlex.quote(code)}"
    limit: Final = max_output_bytes + 1
    capped: Final = (
        f"exec 3> >(head -c {limit}); out=$!; exec 4> >(head -c {limit} >&2); err=$!; "
        f"bash -c {shlex.quote(command)} </dev/null >&3 2>&4; code=$?; exec 3>&- 4>&-; wait $out $err; exit $code"
    )
    return f"bash -c {shlex.quote(capped)}"


class CodeExecutionResult(LiteLLMPydanticObjectBase):
    """Passthrough of the sandbox's own execution output."""

    stdout: str = ""
    stderr: str = ""
    results: list[dict[str, object]] = Field(default_factory=list)
    error: dict[str, object] | None = None
    execution_count: int | None = None
    object: str = "code_execution"

    model_config = {"extra": "allow"}

    _hidden_params: dict[str, builtins.object] = PrivateAttr(default_factory=dict)

    @property
    def hidden_params(self) -> dict[str, builtins.object]:  # mutable-ok: API requires mutation
        return self._hidden_params

    @hidden_params.setter
    def hidden_params(self, hidden_params: dict[str, builtins.object]) -> None:  # mutable-ok: API requires mutation
        self._hidden_params = hidden_params


class BaseSandboxConfig:
    """Provider-agnostic sandbox operations."""

    def _http(self, client: AsyncHTTPHandler | None) -> AsyncHTTPHandler:
        return client or get_async_httpx_client(llm_provider=httpxSpecialProvider.Sandbox)

    def validate_environment(self, api_key: str | None = None, **kwargs: builtins.object) -> str:
        raise NotImplementedError("validate_environment must be implemented by provider")

    async def acreate_sandbox(
        self,
        *,
        template: str | None = None,
        timeout: int | None = None,
        allow_internet_access: bool | None = None,
        api_key: str | None = None,
        **kwargs,
    ) -> ContainerHandle:
        raise NotImplementedError("acreate_sandbox must be implemented by provider")

    async def arun_code(
        self,
        *,
        container: ContainerHandle | str,
        code: str,
        api_key: str | None = None,
        **kwargs,
    ) -> CodeExecutionResult:
        raise NotImplementedError("arun_code must be implemented by provider")

    async def adelete_sandbox(
        self,
        *,
        container: ContainerHandle | str,
        api_key: str | None = None,
        **kwargs,
    ) -> bool:
        raise NotImplementedError("adelete_sandbox must be implemented by provider")

    async def aget_sandbox_cost(
        self,
        *,
        container: ContainerHandle | str,
        api_key: str | None = None,
        api_base: str | None = None,
        client: AsyncHTTPHandler | None = None,
        **kwargs: object,  # kwargs-ok: provider options pass through the shared sandbox entrypoints
    ) -> SandboxCost:
        return estimate_sandbox_cost(container)

    @staticmethod
    def _shell_result(stdout: str, stderr: str, exit_code: int | None, status: str = "ExitCode") -> CodeExecutionResult:
        if max(len(stdout.encode("utf-8")), len(stderr.encode("utf-8"))) > SANDBOX_MAX_OUTPUT_BYTES:
            raise ValueError(
                f"Sandbox output exceeded {SANDBOX_MAX_OUTPUT_BYTES} bytes; aborting to avoid unbounded memory use."
            )
        return CodeExecutionResult.model_validate(
            {
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": exit_code,
                "error": None
                if exit_code == 0
                else {
                    "name": "ExitCode" if exit_code is not None else status,
                    "value": str(exit_code) if exit_code is not None else status,
                    "traceback": stderr,
                },
            }
        )

    async def _read_capped_lines(self, response: httpx.Response) -> list[str]:
        lines: Final[list[str]] = []
        total = 0
        async for line in aiter_sse_lines(response.aiter_text()):
            total += len(line.encode("utf-8"))
            if total > SANDBOX_MAX_OUTPUT_BYTES:
                raise ValueError(
                    f"Sandbox output exceeded {SANDBOX_MAX_OUTPUT_BYTES} bytes; aborting to avoid unbounded memory use."
                )
            lines.append(line)
        return lines
