"""
Base Sandbox transformation configuration.

A sandbox provider runs an executable string inside an isolated container and
returns whatever the sandbox produced. The lifecycle is create container ->
run code -> delete container; `code_interpreter_tool` combines all three.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import Field, PrivateAttr

from litellm.types.llms.base import LiteLLMPydanticObjectBase


class ContainerHandle(LiteLLMPydanticObjectBase):
    """A live sandbox container. Carries everything needed to reach it again."""

    id: str
    provider: str
    domain: Optional[str] = None

    model_config = {"extra": "allow"}

    _hidden_params: dict = PrivateAttr(default_factory=dict)


class CodeExecutionResult(LiteLLMPydanticObjectBase):
    """Passthrough of the sandbox's own execution output."""

    stdout: str = ""
    stderr: str = ""
    results: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[Dict[str, Any]] = None
    execution_count: Optional[int] = None
    object: str = "code_execution"

    model_config = {"extra": "allow"}

    _hidden_params: dict = PrivateAttr(default_factory=dict)


class BaseSandboxConfig:
    """Provider-agnostic sandbox operations."""

    def validate_environment(self, api_key: Optional[str] = None, **kwargs) -> str:
        raise NotImplementedError(
            "validate_environment must be implemented by provider"
        )

    async def acreate_sandbox(
        self,
        *,
        template: Optional[str] = None,
        timeout: Optional[int] = None,
        allow_internet_access: bool = True,
        api_key: Optional[str] = None,
        **kwargs,
    ) -> ContainerHandle:
        raise NotImplementedError("acreate_sandbox must be implemented by provider")

    async def arun_code(
        self,
        *,
        container: Union[ContainerHandle, str],
        code: str,
        api_key: Optional[str] = None,
        **kwargs,
    ) -> CodeExecutionResult:
        raise NotImplementedError("arun_code must be implemented by provider")

    async def adelete_sandbox(
        self,
        *,
        container: Union[ContainerHandle, str],
        api_key: Optional[str] = None,
        **kwargs,
    ) -> bool:
        raise NotImplementedError("adelete_sandbox must be implemented by provider")
