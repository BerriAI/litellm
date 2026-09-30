"""Deep Agents pieces that subclass optional-dependency bases.

Only imported by `litellm.harness.handlers.deepagents_handler.load_deps()`, so `deepagents`,
`langchain` and `langchain-core` are never imported unless Harness.DEEPAGENTS is used.

`SandboxBackend` implements deepagents' `SandboxBackendProtocol` on top of a litellm
`Sandbox`. The agent sees virtual paths rooted at the sandbox workdir (`/src/a.py` is
`<workdir>/src/a.py`); file bytes move through `Sandbox.read/write`, and ls/glob/grep/
delete/execute run plain POSIX commands through `Sandbox.run`, so the same code serves the
local and docker sandboxes (no python3 needed inside the sandbox) and inherits the sandbox's
env scrubbing and path confinement.
"""

from __future__ import annotations

import asyncio
import base64
import posixpath
import re
import shlex
import uuid
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from typing import Any, Final, TypeVar

from deepagents.backends.protocol import (
    DeleteResult,
    EditResult,
    ExecuteResponse,
    FileData,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GlobResult,
    GrepMatch,
    GrepResult,
    LsResult,
    ReadResult,
    SandboxBackendProtocol,
    WriteResult,
)
from deepagents.backends.utils import (
    InvalidGlobPatternError,
    compile_grep_include_glob,
    perform_string_replacement,
    slice_read_response,
)
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse, ToolCallRequest
from langchain.agents.middleware.types import ModelCallResult
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import ToolMessage
from langchain_core.outputs import LLMResult
from langchain_core.tools import BaseTool
from langgraph.types import Command

import litellm
from litellm._logging import verbose_logger
from litellm.constants import HARNESS_SNAPSHOT_SKIP_DIRS
from litellm.harness.context import SessionContext
from litellm.harness.errors import SandboxError
from litellm.harness.sandbox.base import CompletedRun, Sandbox

T = TypeVar("T")

# Module alias so ruff recognises `.exception()` as logging the swallowed error (BLE001).
_logger = verbose_logger
# Models cost_per_token could not price: logged once, then skipped (always 0.0).
_UNPRICED_MODELS: set[str] = set()

DEEPAGENTS_EXECUTE_TIMEOUT_SECONDS: Final = 120.0
DEEPAGENTS_FS_TIMEOUT_SECONDS: Final = 60.0
DEEPAGENTS_MAX_OUTPUT_BYTES: Final = 100_000
_EXIT_NOT_FOUND: Final = 3
_EXIT_NOT_DIR: Final = 4
_EXIT_TIMEOUT: Final = 124
_READ_ONLY_ERROR: Final = "Error: this session is read-only; files cannot be changed"
_NO_EXECUTE_ERROR: Final = "Error: shell execution is disabled for this session"
# $1 = directory. Prints "d/<name>" or "f/<name>" per entry ("/" never appears in a name).
_LS_SCRIPT: Final = (
    '[ -e "$1" ] || exit 3; [ -d "$1" ] || exit 4; cd "$1" || exit 5; '
    'for f in * .[!.]* ..?*; do if [ -e "$f" ] || [ -L "$f" ]; then '
    'if [ -d "$f" ]; then printf "d/%s\\n" "$f"; else printf "f/%s\\n" "$f"; fi; fi; done'
)
_DELETE_SCRIPT: Final = '[ -e "$1" ] || [ -L "$1" ] || exit 3; rm -rf -- "$1"'
# $1 = path. Prints its fully symlink-resolved absolute path (the path itself may not exist
# yet, e.g. a file about to be written; then its parent directory is resolved).
_REALPATH_SCRIPT: Final = (
    'p="$1"; if [ -e "$p" ] || [ -L "$p" ]; then realpath -- "$p"; '
    'else d=$(realpath -- "$(dirname -- "$p")") && printf "%s/%s\\n" "$d" "$(basename -- "$p")"; fi'
)
_GREP_LINE: Final = re.compile(r"^(.+?):(\d+):(.*)$")
_FILTER_MIDDLEWARE_NAME: Final = "LiteLLMHarnessToolFilter"


def _decode(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


class SandboxBackend(SandboxBackendProtocol):
    """deepagents backend whose files and shell live in a litellm Sandbox."""

    def __init__(
        self,
        sandbox: Sandbox,
        *,
        loop: asyncio.AbstractEventLoop,
        writable: bool = True,
        allow_execute: bool = True,
    ) -> None:
        self._sandbox = sandbox
        self._loop = loop
        self._root = posixpath.normpath(sandbox.workdir)
        self._real_root: str | None = None
        self._writable = writable
        self._allow_execute = allow_execute
        self._id = f"litellm-harness-{uuid.uuid4().hex[:8]}"

    @property
    def id(self) -> str:
        return self._id

    # -- paths --------------------------------------------------------------

    def to_real(self, path: str) -> str:
        """Sandbox path for a virtual path (or an absolute path already under workdir)."""
        normalized = posixpath.normpath("/" + path.lstrip("/"))
        if ".." in normalized.split("/"):
            raise ValueError(f"path traversal not allowed: {path}")
        if normalized == self._root or normalized.startswith(self._root + "/"):
            return normalized
        if normalized == "/":
            return self._root
        return self._root + normalized

    async def to_confined(self, path: str) -> str:
        """to_real, then resolve symlinks inside the sandbox and refuse anything outside workdir.

        A repo can contain `link -> ~/.aws/credentials`; without this, read/grep/glob would
        follow it and read host secrets even in read-only mode.
        """
        real = self.to_real(path)
        done = await self._run(["sh", "-c", _REALPATH_SCRIPT, "sh", real])
        resolved = done.stdout.strip()
        if done.exit_code != 0 or not resolved:
            raise ValueError(f"path not found: {path}")
        root = await self._resolved_root()
        if resolved != root and not resolved.startswith(root + "/"):
            raise ValueError(f"path resolves outside the workspace: {path}")
        return real

    async def _resolved_root(self) -> str:
        if self._real_root is None:
            done = await self._run(["realpath", "--", self._root])
            self._real_root = done.stdout.strip() if done.exit_code == 0 and done.stdout.strip() else self._root
        return self._real_root

    def to_virtual(self, real: str) -> str:
        if real == self._root:
            return "/"
        if real.startswith(self._root + "/"):
            return real[len(self._root) :]
        return real

    # -- sync bridge (deepagents only calls these outside the event loop) ---

    def _sync(self, coro: Coroutine[Any, Any, T]) -> T:
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self._loop:
            coro.close()
            raise RuntimeError("SandboxBackend sync methods cannot run on the event loop thread")
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    async def _run(self, cmd: list[str], timeout: float | None = DEEPAGENTS_FS_TIMEOUT_SECONDS) -> CompletedRun:
        return await self._sandbox.run(cmd, timeout=timeout)

    # -- ls -----------------------------------------------------------------

    async def als(self, path: str) -> LsResult:
        try:
            real = await self.to_confined(path)
            done = await self._run(["sh", "-c", _LS_SCRIPT, "sh", real])
        except (ValueError, SandboxError) as e:
            return LsResult(error=f"Path '{path}': {e}")
        if done.exit_code == _EXIT_NOT_FOUND:
            return LsResult(error=f"Path '{path}': path_not_found")
        if done.exit_code == _EXIT_NOT_DIR:
            return LsResult(error=f"Path '{path}': not_a_directory")
        if done.exit_code != 0:
            return LsResult(error=f"Path '{path}': {done.stderr.strip() or 'ls failed'}")
        base = self.to_virtual(real).rstrip("/")
        entries: list[FileInfo] = []
        for line in done.stdout.splitlines():
            kind, _, name = line.partition("/")
            if not name:
                continue
            is_dir = kind == "d"
            entries.append({"path": f"{base}/{name}" + ("/" if is_dir else ""), "is_dir": is_dir})
        entries.sort(key=lambda e: e["path"])
        return LsResult(entries=entries)

    def ls(self, path: str) -> LsResult:
        return self._sync(self.als(path))

    # -- read / write / edit ------------------------------------------------

    async def _read_bytes(self, path: str) -> bytes:
        return await self._sandbox.read(await self.to_confined(path))

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        try:
            data = await self._read_bytes(file_path)
        except ValueError as e:
            return ReadResult(error=f"Error reading file '{file_path}': {e}")
        except SandboxError:
            return ReadResult(error=f"File '{file_path}' not found")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            encoded = base64.standard_b64encode(data).decode("ascii")
            return ReadResult(file_data=FileData(content=encoded, encoding="base64"))
        return slice_read_response(FileData(content=text, encoding="utf-8"), offset, limit)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        return self._sync(self.aread(file_path, offset, limit))

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        if not self._writable:
            return WriteResult(error=_READ_ONLY_ERROR)
        try:
            await self._sandbox.write(await self.to_confined(file_path), content.encode("utf-8"))
        except (ValueError, SandboxError) as e:
            return WriteResult(error=f"Error writing file '{file_path}': {e}")
        return WriteResult(path=file_path)

    def write(self, file_path: str, content: str) -> WriteResult:
        return self._sync(self.awrite(file_path, content))

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        if not self._writable:
            return EditResult(error=_READ_ONLY_ERROR)
        try:
            content = _decode(await self._read_bytes(file_path))
        except ValueError as e:
            return EditResult(error=f"Error editing file '{file_path}': {e}")
        except SandboxError:
            return EditResult(error=f"Error: File '{file_path}' not found")
        old = old_string.replace("\r\n", "\n")
        new = new_string.replace("\r\n", "\n")
        replaced = perform_string_replacement(content.replace("\r\n", "\n"), old, new, replace_all)
        if isinstance(replaced, str):
            return EditResult(error=replaced)
        new_content, occurrences = replaced
        try:
            await self._sandbox.write(await self.to_confined(file_path), new_content.encode("utf-8"))
        except SandboxError as e:
            return EditResult(error=f"Error editing file '{file_path}': {e}")
        return EditResult(path=file_path, occurrences=int(occurrences))

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        return self._sync(self.aedit(file_path, old_string, new_string, replace_all))

    async def adelete(self, file_path: str) -> DeleteResult:
        if not self._writable:
            return DeleteResult(error=_READ_ONLY_ERROR)
        try:
            real = await self.to_confined(file_path)
            if real == self._root:
                return DeleteResult(error="Error: refusing to delete the workspace root")
            done = await self._run(["sh", "-c", _DELETE_SCRIPT, "sh", real])
        except (ValueError, SandboxError) as e:
            return DeleteResult(error=f"Error deleting '{file_path}': {e}")
        if done.exit_code == _EXIT_NOT_FOUND:
            return DeleteResult(error=f"Error: '{file_path}' not found")
        if done.exit_code != 0:
            return DeleteResult(error=f"Error deleting '{file_path}': {done.stderr.strip()}")
        return DeleteResult(path=file_path)

    def delete(self, file_path: str) -> DeleteResult:
        return self._sync(self.adelete(file_path))

    # -- glob / grep --------------------------------------------------------

    def _find_cmd(self, root: str) -> list[str]:
        prune: list[str] = []
        for name in sorted(HARNESS_SNAPSHOT_SKIP_DIRS):
            prune += ["-o", "-name", name] if prune else ["-name", name]
        # -P: never follow symlinks, so a repo link to ~/.aws cannot pull host files in.
        return ["find", "-P", root, "(", *prune, ")", "-prune", "-o", "-type", "f", "-print"]

    def _grep_cmd(self, pattern: str, root: str) -> list[str]:
        # grep only the regular files `find -P -type f` lists: symlinks are never followed,
        # whatever grep implementation (GNU -R vs BSD -r) the sandbox has.
        find_cmd = " ".join(shlex.quote(part) for part in self._find_cmd(root))
        return ["sh", "-c", f'{find_cmd} | tr "\\n" "\\0" | xargs -0 grep -nHFI -e "$1" --', "sh", pattern]

    async def aglob(self, pattern: str, path: str | None = None) -> GlobResult:
        try:
            matcher = compile_grep_include_glob(pattern)
            root = await self.to_confined(path or "/")
            done = await self._run(self._find_cmd(root))
        except (InvalidGlobPatternError, ValueError, SandboxError) as e:
            return GlobResult(error=str(e), matches=None)
        if done.exit_code != 0 and not done.stdout:
            return GlobResult(matches=[])
        matches: list[FileInfo] = []
        for real in done.stdout.splitlines():
            rel = posixpath.relpath(real, root)
            if matcher(rel):
                matches.append({"path": self.to_virtual(real), "is_dir": False})
        matches.sort(key=lambda m: m["path"])
        return GlobResult(matches=matches, truncated=done.exit_code != 0)

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        return self._sync(self.aglob(pattern, path))

    async def agrep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        *,
        max_count: int | None = None,
    ) -> GrepResult:
        try:
            include = compile_grep_include_glob(glob) if glob else None
            root = await self.to_confined(path or "/")
            done = await self._run(self._grep_cmd(pattern, root))
        except (InvalidGlobPatternError, ValueError, SandboxError) as e:
            return GrepResult(error=f"Path '{path or '/'}': {e}")
        if done.exit_code not in (0, 1) and not done.stdout:
            return GrepResult(error=f"Path '{path or '/'}': {done.stderr.strip() or 'grep failed'}")
        matches: list[GrepMatch] = []
        for line in done.stdout.splitlines():
            parsed = _GREP_LINE.match(line)
            if parsed is None:
                continue
            real, line_no, text = parsed.group(1), int(parsed.group(2)), parsed.group(3)
            if include is not None and not include(posixpath.relpath(real, root)):
                continue
            matches.append({"path": self.to_virtual(real), "line": line_no, "text": text})
        if max_count is not None and len(matches) > max_count:
            return GrepResult(matches=matches[:max_count], truncated=True)
        return GrepResult(matches=matches)

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        *,
        max_count: int | None = None,
    ) -> GrepResult:
        return self._sync(self.agrep(pattern, path, glob, max_count=max_count))

    # -- upload / download --------------------------------------------------

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        responses: list[FileUploadResponse] = []
        for path, data in files:
            if not self._writable:
                responses.append(FileUploadResponse(path=path, error="permission_denied"))
                continue
            try:
                await self._sandbox.write(await self.to_confined(path), data)
                responses.append(FileUploadResponse(path=path))
            except ValueError:
                responses.append(FileUploadResponse(path=path, error="invalid_path"))
            except SandboxError as e:
                responses.append(FileUploadResponse(path=path, error=str(e)))
        return responses

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return self._sync(self.aupload_files(files))

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        responses: list[FileDownloadResponse] = []
        for path in paths:
            try:
                responses.append(FileDownloadResponse(path=path, content=await self._read_bytes(path)))
            except ValueError:
                responses.append(FileDownloadResponse(path=path, error="invalid_path"))
            except SandboxError:
                responses.append(FileDownloadResponse(path=path, error="file_not_found"))
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        return self._sync(self.adownload_files(paths))

    # -- execute ------------------------------------------------------------

    async def aexecute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        if not self._allow_execute:
            return ExecuteResponse(output=_NO_EXECUTE_ERROR, exit_code=1)
        limit = float(timeout) if timeout else DEEPAGENTS_EXECUTE_TIMEOUT_SECONDS
        try:
            done = await self._run(["sh", "-c", command], timeout=limit)
        except SandboxError as e:
            return ExecuteResponse(output=f"Error: {e}", exit_code=_EXIT_TIMEOUT)
        output = done.stdout
        if done.stderr:
            output = f"{output}\n{done.stderr}" if output else done.stderr
        truncated = len(output.encode("utf-8")) > DEEPAGENTS_MAX_OUTPUT_BYTES
        if truncated:
            output = output.encode("utf-8")[:DEEPAGENTS_MAX_OUTPUT_BYTES].decode("utf-8", errors="ignore")
        return ExecuteResponse(output=output, exit_code=done.exit_code, truncated=truncated)

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        return self._sync(self.aexecute(command, timeout=timeout))


def _tool_name(tool: BaseTool | Mapping[str, object]) -> str | None:
    name = tool.get("name") if isinstance(tool, Mapping) else tool.name
    return name if isinstance(name, str) else None


def _blocked_message(request: ToolCallRequest, blocked: frozenset[str]) -> ToolMessage | None:
    name = request.tool_call["name"]
    if name not in blocked:
        return None
    return ToolMessage(
        content=f"Error: {name} is disabled for this session.",
        tool_call_id=request.tool_call["id"] or "",
        name=name,
        status="error",
    )


class ToolFilterMiddleware(AgentMiddleware):
    """Hide tools from the model and refuse calls to them (disable_tools / permissions)."""

    def __init__(self, blocked: frozenset[str]) -> None:
        super().__init__()
        self._blocked = blocked

    @property
    def name(self) -> str:
        return _FILTER_MIDDLEWARE_NAME

    def _filtered(self, request: ModelRequest) -> ModelRequest:
        return request.override(tools=[t for t in request.tools if _tool_name(t) not in self._blocked])

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelCallResult:
        return handler(self._filtered(request))

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> ModelCallResult:
        return await handler(self._filtered(request))

    def wrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command]
    ) -> ToolMessage | Command:
        return _blocked_message(request, self._blocked) or handler(request)

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]]
    ) -> ToolMessage | Command:
        return _blocked_message(request, self._blocked) or await handler(request)


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def message_cost(message: object, cost_model: str | None, input_tokens: int, output_tokens: int) -> float:
    """Cost of one model call: response_cost reported by litellm, else cost_per_token, else 0."""
    metadata = getattr(message, "response_metadata", None) or {}
    reported = _number(metadata.get("response_cost"))
    if reported is not None:
        return reported
    if not cost_model or cost_model in _UNPRICED_MODELS:
        return 0.0
    try:
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=cost_model,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
        )
        return float(prompt_cost) + float(completion_cost)
    except Exception:  # litellm raises plain Exception for unmapped models
        _UNPRICED_MODELS.add(cost_model)
        _logger.exception("harness deepagents: no cost for %s; counting its calls as 0", cost_model)
        return 0.0


def record_llm_usage(ctx: SessionContext, cost_model: str | None, response: LLMResult) -> None:
    """Add one model call's tokens and cost to the session counters. Never raises."""
    try:
        ctx.calls += 1
        for generations in response.generations:
            for generation in generations:
                message = getattr(generation, "message", None)
                usage = getattr(message, "usage_metadata", None) or {}
                input_tokens = int(usage.get("input_tokens") or 0)
                output_tokens = int(usage.get("output_tokens") or 0)
                ctx.input_tokens += input_tokens
                ctx.output_tokens += output_tokens
                ctx.cost += message_cost(message, cost_model, input_tokens, output_tokens)
    except Exception:
        # Usage accounting must never fail a turn; log with traceback and move on.
        _logger.exception("harness deepagents: usage accounting failed")


class UsageCallback(AsyncCallbackHandler):
    """Counts every model call in the graph, subagents and summarization included."""

    def __init__(self, ctx: SessionContext, cost_model: str | None) -> None:
        self._ctx = ctx
        self._cost_model = cost_model

    async def on_llm_end(self, response: LLMResult, **kwargs: object) -> None:
        record_llm_usage(self._ctx, self._cost_model, response)
