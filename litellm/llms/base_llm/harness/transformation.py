"""
Base agent-harness transformation configuration.

A harness is a complete agent runtime (Claude Code, Codex, OpenCode, Deep Agents).
Like the LLM provider configs in `litellm/llms/base_llm/chat/transformation.py`, a
harness config only translates: LiteLLM's session parameters in, the runtime's native
command / config / event stream out. It never does I/O. A handler in
`litellm/harness/handlers/` owns the sandbox, the process and the per-session model
endpoint, and calls these transforms.

Adding a CLI harness is one subclass of `BaseCLIHarnessConfig` in
`litellm/llms/<harness>/harness/transformation.py`, plus one line in
`ProviderConfigManager.get_provider_harness_config`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar

from litellm.harness.errors import HarnessError, OptionsMismatch
from litellm.harness.types import Capabilities, Event, Harness

if TYPE_CHECKING:
    from litellm.harness.context import SessionContext

# A config's typed options (ClaudeCodeOptions, CodexOptions, ...) and its per-turn parser state.
OptionsT = TypeVar("OptionsT")
StreamStateT = TypeVar("StreamStateT")


def event_list(*events: Event) -> Sequence[Event]:
    """A transform_stream_line result. One place builds it so every parser returns the same shape."""
    return list(events)  # mutable-ok: stream-line results are list-shaped; callers and tests compare with list literals


class HarnessTurnError(HarnessError):
    """The runtime reported a failed turn. The runtime maps this to stop_reason='runtime_error'."""


@dataclass(frozen=True)
class HarnessSessionSetup:
    """What the handler must prepare in the sandbox before the first turn.

    Paths are relative to `private_dir` (a per-session temp dir inside the sandbox)
    unless they are absolute.
    """

    files: Mapping[str, bytes] = field(default_factory=dict)
    # (dir inside private_dir, cache subpath under ~/.cache/litellm-harness) linked so a
    # later session can resume the runtime's own conversation.
    persisted_dirs: Sequence[tuple[str, str]] = ()
    # Where skill folders are copied, relative to private_dir, or absolute.
    skills_dir: str | None = None
    # Env passed on every turn. Values may contain `{private_dir}`.
    env: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class HarnessTurnRequest:
    """One turn of a CLI runtime: the process to run and what to send on stdin."""

    argv: Sequence[str]
    env: Mapping[str, str]
    stdin: str
    cwd: str | None = None


@dataclass(frozen=True)
class HarnessTurnResponse:
    """What the runtime produced for one turn, after the process exited."""

    final_text: str
    output_json: str | None = None


class BaseHarnessConfig(ABC, Generic[OptionsT]):
    """Declares what a harness is and validates a session before anything starts."""

    harness: ClassVar[Harness]
    options_type: type[OptionsT]
    capabilities: ClassVar[Capabilities]
    # CLI runtimes call a per-session model endpoint; in-process ones call LiteLLM directly.
    uses_model_endpoint: ClassVar[bool] = True

    def get_options(self, ctx: SessionContext) -> OptionsT:
        """ctx.options, or this harness's default options."""
        options = ctx.options
        if options is None:
            return self.options_type()
        if not isinstance(options, self.options_type):
            raise OptionsMismatch(
                f"{type(options).__name__} cannot be used with Harness.{self.harness.name}; "
                f"use {self.options_type.__name__}"
            )
        return options

    def validate_environment(self, ctx: SessionContext) -> None:
        """Static checks on the session. Raise OptionsMismatch / ValueError early."""
        self.get_options(ctx)


class BaseCLIHarnessConfig(BaseHarnessConfig[OptionsT], Generic[OptionsT, StreamStateT]):
    """A runtime driven as a subprocess that prints one JSON event per line."""

    @abstractmethod
    def get_binary(self) -> str:
        """Executable that must be on the sandbox's PATH."""

    @abstractmethod
    def get_install_hint(self) -> str:
        """How to install the binary; shown in HarnessInstallFailed."""

    @abstractmethod
    def transform_session_setup(self, ctx: SessionContext, private_dir: str) -> HarnessSessionSetup:
        """Config files, env and persisted dirs for the session."""

    @abstractmethod
    def transform_turn_request(
        self,
        ctx: SessionContext,
        setup: HarnessSessionSetup,
        private_dir: str,
        prompt: str,
        native_session_id: str | None,
    ) -> HarnessTurnRequest:
        """argv / env / stdin for one turn. native_session_id is set after the first turn."""

    @abstractmethod
    def create_stream_state(self) -> StreamStateT:
        """Fresh per-turn parser state."""

    @abstractmethod
    def transform_stream_line(self, line: Mapping[str, object], state: StreamStateT) -> Sequence[Event]:
        """One decoded JSON line from stdout to zero or more events. Pure."""

    @abstractmethod
    def get_native_session_id(self, state: StreamStateT) -> str | None:
        """The runtime's own session / thread id, once the stream has reported it."""

    @abstractmethod
    def transform_turn_response(
        self,
        ctx: SessionContext,
        state: StreamStateT,
        exit_code: int,
        stderr_tail: Sequence[str],
    ) -> HarnessTurnResponse:
        """Final text and structured output, or raise HarnessTurnError."""
