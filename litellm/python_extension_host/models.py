from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from litellm.python_extension.generated.v1 import extension_host_pb2 as pb


@dataclass(frozen=True, slots=True)
class ExtensionConfig:
    id: str
    kind: pb.ExtensionKind
    entrypoint: str
    constructor_json: bytes


@dataclass(frozen=True, slots=True)
class LoadedExtension:
    config: ExtensionConfig
    target: object
    hooks: tuple[str, ...]
    callable_target: bool
    async_callable: bool

    def descriptor(self) -> pb.ExtensionDescriptor:
        return pb.ExtensionDescriptor(
            id=self.config.id,
            kind=self.config.kind,
            hooks=self.hooks,
            callable=self.callable_target,
            async_callable=self.async_callable,
        )


def operation_ok() -> pb.OperationResult:
    return pb.OperationResult(ok=True)


def operation_error(code: pb.ErrorCode, message: str) -> pb.OperationResult:
    return pb.OperationResult(ok=False, error_code=code, error_message=message)


CALLBACK_KIND: Final = pb.EXTENSION_KIND_CALLBACK
GUARDRAIL_KIND: Final = pb.EXTENSION_KIND_GUARDRAIL
