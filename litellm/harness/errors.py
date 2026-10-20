"""Exceptions raised by litellm.harness."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from litellm.harness.types import Result


class HarnessError(Exception):
    """Base class for every litellm.harness error."""


class CapabilityUnsupported(HarnessError):
    """The harness cannot do what was asked. Raised before the runtime starts."""


class OptionsMismatch(HarnessError):
    """Options for a different harness, or a native option LiteLLM manages itself."""


class HarnessInstallFailed(HarnessError):
    """The runtime is missing from the sandbox or failed to start."""


class SandboxError(HarnessError):
    """The sandbox failed to start, run a command, or reach the host."""


class SessionClosed(HarnessError):
    """A turn was started on a session that is closed or detached."""


class StateIncompatible(HarnessError):
    """resume() was given a State from another harness or an unreadable version."""


class OutputInvalid(HarnessError):
    """The final answer did not validate against output=."""

    def __init__(self, message: str, raw: str, result: Result | None = None) -> None:
        super().__init__(message)
        self.raw = raw
        self.result = result
