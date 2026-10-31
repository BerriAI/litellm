"""PrizedSandbox: run harness commands and files in a Prized sandbox (a Firecracker microVM) over its HTTP API."""

from __future__ import annotations

import asyncio
import math
import posixpath
import shlex
import uuid
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from litellm.harness.errors import CapabilityUnsupported, SandboxError
from litellm.harness.sandbox.base import CompletedRun
from litellm.harness.sandbox.snapshot import parse_sha256sum, remote_snapshot_script
from litellm.secret_managers.main import get_secret_str

PRIZED_API_BASE: Final = "https://api.prized.dev/api/v1"
PRIZED_DEFAULT_TEMPLATE: Final = "prized-base-v1"
PRIZED_HOME: Final = "/home/prized"
PRIZED_WORKDIR: Final = "/home/prized/work"
PRIZED_EXEC_MAX_SECONDS: Final = 3600
_START_TIMEOUT_SECONDS: Final = 180.0
_READY_TIMEOUT_SECONDS: Final = 30.0
_POLL_SECONDS: Final = 0.5
_ANSWER_SLACK_SECONDS: Final = 30.0
_STOP_TIMEOUT_SECONDS: Final = 15.0
_RETRYABLE_EDGE_STATUSES: Final = frozenset({502, 503})
_RUN_DIR: Final = "/tmp/litellm-harness-runs"
_WHICH_SCRIPT: Final = 'command -v "$1"'
_STOP_SCRIPT: Final = (
    'f="$1"; i=0; while [ ! -s "$f" ] && [ $i -lt 20 ]; do sleep 0.1; i=$((i+1)); done; '
    '[ -s "$f" ] || exit 0; pg=$(cat "$f"); kill -TERM -- "-$pg" 2>/dev/null; i=0; '
    'while kill -0 -- "-$pg" 2>/dev/null && [ $i -lt 20 ]; do sleep 0.1; i=$((i+1)); done; '
    'kill -KILL -- "-$pg" 2>/dev/null; rm -f "$f"'
)


class _SandboxDoc(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore", populate_by_name=True)

    id: str
    state: str
    toolbox_base: str | None = Field(default=None, alias="toolboxBase")
    error_reason: str | None = Field(default=None, alias="errorReason")


class _SandboxEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    sandbox: _SandboxDoc


class _ExecAnswer(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore", populate_by_name=True)

    stdout: str = ""
    stderr: str = ""
    exit_code: int = Field(alias="exitCode")
    timed_out: bool = Field(default=False, alias="timedOut")


class _PrizedProcess:
    """A command run to completion in the sandbox; its output arrives on the readers when it exits."""

    def __init__(self, run: asyncio.Task[CompletedRun]) -> None:
        self.stdin: asyncio.StreamWriter | None = None
        self.stdout: Final = asyncio.StreamReader()
        self.stderr: Final = asyncio.StreamReader()
        self._run: Final = run
        self.done: Final = run
        run.add_done_callback(self._deliver)

    def _deliver(self, run: asyncio.Task[CompletedRun]) -> None:
        if not run.cancelled() and run.exception() is None:
            self.stdout.feed_data(run.result().stdout.encode())
            self.stderr.feed_data(run.result().stderr.encode())
        self.stdout.feed_eof()
        self.stderr.feed_eof()

    async def wait(self) -> int:
        try:
            return (await asyncio.shield(self._run)).exit_code
        except asyncio.CancelledError:
            if self._run.cancelled():
                return -9
            raise

    async def kill(self) -> None:
        self._run.cancel()
        await asyncio.gather(self._run, return_exceptions=True)


class PrizedSandbox:
    """Sandbox in a Prized microVM, created lazily on first use and deleted on close().

    Pass `sandbox_id` to attach to a sandbox you already have instead; close() then leaves it running.
    Commands run under `bash` as the user `prized`, with stdin at EOF and output returned when they exit.
    Files move through the sandbox's file API, so `workdir` and every path must be under /home/prized.
    The sandbox cannot reach this machine, so CLI harnesses that call back to a local model endpoint
    (Claude Code, Codex, OpenCode) are not supported; Deep Agents and Tool Loop are.
    """

    is_container = True

    def __init__(
        self,
        *,
        template: str = PRIZED_DEFAULT_TEMPLATE,
        sandbox_id: str | None = None,
        workdir: str = PRIZED_WORKDIR,
        env: Mapping[str, str] | None = None,
        labels: Mapping[str, str] | None = None,
        vcpu: int | None = None,
        memory_mb: int | None = None,
        max_runtime_min: int | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        normalized: Final = posixpath.normpath(workdir)
        if not posixpath.isabs(normalized) or not _under_home(normalized):
            raise SandboxError(f"prized workdir must be an absolute path under {PRIZED_HOME}: {workdir}")
        self.workdir: str = normalized
        self.template = template
        self.env: Mapping[str, str] = MappingProxyType(dict(env or ()))
        self.labels: Mapping[str, str] = MappingProxyType(dict(labels or ()))
        self.sandbox_id: str | None = sandbox_id
        self._owns_sandbox = sandbox_id is None
        self._resources: Final = MappingProxyType(
            {k: v for k, v in (("vcpu", vcpu), ("memoryMb", memory_mb)) if v is not None}
        )
        self._max_runtime_min = max_runtime_min
        self._api_key = api_key
        self._api_base = (api_base or PRIZED_API_BASE).rstrip("/")
        self._client = client
        self._owns_client = False
        self._toolbox: str | None = None
        self._start_lock: Final = asyncio.Lock()
        self._processes: Final[set[_PrizedProcess]] = set()  # mutable-ok: live-process registry, killed on close()
        self._closed = False

    def __repr__(self) -> str:
        return f"PrizedSandbox({self.sandbox_id or self.template!r}, workdir={self.workdir!r})"

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient()
            self._owns_client = True
        return self._client

    def _headers(self) -> Mapping[str, str]:
        key: Final = self._api_key or get_secret_str("PRIZED_API_KEY")
        if not key:
            raise SandboxError("Prized API key not set. Set PRIZED_API_KEY or pass api_key=...")
        return {"Authorization": f"Bearer {key}"}

    def sandbox_path(self, path: str) -> str:
        """Absolute sandbox path; relative paths resolve against workdir."""
        return posixpath.normpath(path if posixpath.isabs(path) else posixpath.join(self.workdir, path))

    async def start(self) -> str:
        """Create (or attach to) the sandbox if needed, wait until it answers commands, and return its id."""
        if self._closed:
            raise SandboxError("sandbox is closed")
        async with self._start_lock:
            if self.sandbox_id is not None and self._toolbox is not None:
                return self.sandbox_id
            doc: Final = await (self._get(self.sandbox_id) if self.sandbox_id else self._create())
            self.sandbox_id = doc.id
            running: Final = await self._wait_running(doc)
            if running.toolbox_base is None:
                raise SandboxError(f"prized sandbox {doc.id} has no command endpoint")
            self._toolbox = running.toolbox_base.rstrip("/")
            await self._wait_ready()
            if self._owns_sandbox:
                await self._run_line(f"mkdir -p -- {shlex.quote(self.workdir)}", cwd=PRIZED_HOME, timeout=60)
            return doc.id

    async def _create(self) -> _SandboxDoc:
        body: Final = {
            "template": self.template,
            "labels": {"created-by": "litellm-harness", **self.labels},
            **({"envVars": dict(self.env)} if self.env else {}),
            **({"resources": dict(self._resources)} if self._resources else {}),
            **({"maxRuntimeMin": self._max_runtime_min} if self._max_runtime_min else {}),
        }
        response: Final = await self._http().post(
            f"{self._api_base}/sandboxes",
            params={"wait": "running"},
            json=body,
            headers={**self._headers(), "Idempotency-Key": f"litellm-{uuid.uuid4().hex}"},
            timeout=_START_TIMEOUT_SECONDS,
        )
        return _sandbox_doc(response, "create")

    async def _get(self, sandbox_id: str) -> _SandboxDoc:
        response: Final = await self._http().get(
            f"{self._api_base}/sandboxes/{sandbox_id}", headers=self._headers(), timeout=_READY_TIMEOUT_SECONDS
        )
        return _sandbox_doc(response, f"get {sandbox_id}")

    async def _wait_running(self, doc: _SandboxDoc) -> _SandboxDoc:
        deadline: Final = asyncio.get_running_loop().time() + _START_TIMEOUT_SECONDS
        current = doc  # rebind-ok: polled until the sandbox settles
        while current.state != "running":
            if current.state in ("deleting", "deleted", "error"):
                raise SandboxError(f"prized sandbox {doc.id} is {current.state}: {current.error_reason or ''}".strip())
            if current.state == "paused":
                return current
            if asyncio.get_running_loop().time() > deadline:
                raise SandboxError(f"prized sandbox {doc.id} did not start within {_START_TIMEOUT_SECONDS:g}s")
            await asyncio.sleep(_POLL_SECONDS)
            current = await self._get(doc.id)
        return current

    async def _wait_ready(self) -> None:
        """The edge can answer 502 for a moment after a sandbox reaches running; a harmless `true` waits it out."""
        deadline: Final = asyncio.get_running_loop().time() + _READY_TIMEOUT_SECONDS
        while True:
            response = await self._post_execute({"command": "true", "timeout": 10}, timeout=40.0)
            if response.status_code not in _RETRYABLE_EDGE_STATUSES:
                _exec_answer(response)
                return
            if asyncio.get_running_loop().time() > deadline:
                raise SandboxError(f"prized sandbox {self.sandbox_id} did not answer commands ({response.status_code})")
            await asyncio.sleep(_POLL_SECONDS)

    async def _post_execute(self, body: Mapping[str, object], *, timeout: float) -> httpx.Response:
        return await self._http().post(
            f"{self._toolbox}/process/execute", json=dict(body), headers=self._headers(), timeout=timeout
        )

    async def _run_line(
        self, line: str, *, cwd: str, env: Mapping[str, str] | None = None, timeout: int = PRIZED_EXEC_MAX_SECONDS
    ) -> _ExecAnswer:
        body: Final = {"command": line, "cwd": cwd, "timeout": timeout, **({"env": dict(env)} if env else {})}
        return _exec_answer(await self._post_execute(body, timeout=timeout + _ANSWER_SLACK_SECONDS))

    async def exec(
        self,
        cmd: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
    ) -> _PrizedProcess:
        if not cmd:
            raise SandboxError("exec() needs a non-empty command")
        await self.start()
        process: Final = _PrizedProcess(asyncio.ensure_future(self.run(cmd, env=env, cwd=cwd)))
        self._processes.add(process)
        process.done.add_done_callback(lambda _: self._processes.discard(process))
        return process

    async def run(
        self,
        cmd: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        timeout: float | None = None,
    ) -> CompletedRun:
        if not cmd:
            raise SandboxError("run() needs a non-empty command")
        await self.start()
        limit: Final = PRIZED_EXEC_MAX_SECONDS if timeout is None else max(1, math.ceil(timeout))
        if limit > PRIZED_EXEC_MAX_SECONDS:
            raise SandboxError(f"prized runs one command for at most {PRIZED_EXEC_MAX_SECONDS}s")
        pid_file: Final = f"{_RUN_DIR}/{uuid.uuid4().hex}.pid"
        line: Final = (
            f'mkdir -p {_RUN_DIR} && printf %s "$$" > {pid_file}; '
            f"{shlex.join(cmd)}; status=$?; rm -f {pid_file}; exit $status"
        )
        try:
            answer: Final = await self._run_line(
                line, cwd=self.sandbox_path(cwd or self.workdir), env={**self.env, **(env or {})}, timeout=limit
            )
        except BaseException:
            await asyncio.shield(self._stop(pid_file))
            raise
        if answer.timed_out:
            raise SandboxError(f"command timed out after {timeout}s: {cmd[0]}")
        return CompletedRun(stdout=answer.stdout, stderr=answer.stderr, exit_code=answer.exit_code)

    async def _stop(self, pid_file: str) -> None:
        """Kill the process group of a command whose call was cancelled or lost; best effort."""
        try:
            await asyncio.wait_for(
                self._run_line(shlex.join(("sh", "-c", _STOP_SCRIPT, "sh", pid_file)), cwd=PRIZED_HOME, timeout=10),
                _STOP_TIMEOUT_SECONDS,
            )
        except (SandboxError, httpx.HTTPError, asyncio.TimeoutError):
            return

    async def read(self, path: str) -> bytes:
        await self.start()
        target: Final = self.sandbox_path(path)
        response: Final = await self._http().get(
            f"{self._toolbox}/files/download", params={"path": target}, headers=self._headers(), timeout=600.0
        )
        if response.status_code != 200:
            raise SandboxError(f"could not read {target}: {_edge_error(response)}")
        return response.content

    async def write(self, path: str, data: bytes) -> None:
        await self.start()
        target: Final = self.sandbox_path(path)
        response: Final = await self._http().post(
            f"{self._toolbox}/files/upload",
            params={"path": target},
            content=data,
            headers={**self._headers(), "Content-Type": "application/octet-stream"},
            timeout=600.0,
        )
        if response.status_code != 200:
            raise SandboxError(f"could not write {target}: {_edge_error(response)}")

    def host_url(self, port: int) -> str:
        raise CapabilityUnsupported(
            "a Prized sandbox cannot reach this machine, so CLI harnesses (Claude Code, Codex, OpenCode) "
            "cannot call the local model endpoint from it; use Harness.DEEPAGENTS or Harness.TOOL_LOOP"
        )

    async def which(self, binary: str) -> str | None:
        found: Final = await self.run(("sh", "-lc", _WHICH_SCRIPT, "sh", binary), timeout=30)
        path: Final = found.stdout.strip()
        return path if found.exit_code == 0 and path else None

    async def tempdir(self) -> str:
        """A fresh directory under /home/prized, so the file API reaches it."""
        made: Final = await self.run(("mktemp", "-d", "-p", PRIZED_HOME, "litellm-harness-XXXXXX"), timeout=30)
        path: Final = made.stdout.strip()
        if made.exit_code != 0 or not path:
            raise SandboxError(f"mktemp -d failed: {made.stderr.strip()}")
        return path

    async def snapshot(self) -> Mapping[str, str]:
        listed: Final = await self.run(("sh", "-c", remote_snapshot_script(), "sh", self.workdir), timeout=300)
        if listed.exit_code != 0:
            raise SandboxError(f"snapshot failed: {listed.stderr.strip()}")
        return parse_sha256sum(listed.stdout)

    async def close(self) -> None:
        """Stop running commands and delete the sandbox this object created; an attached one keeps running."""
        if self._closed:
            return
        self._closed = True
        await asyncio.gather(*(process.kill() for process in self._processes), return_exceptions=True)
        self._processes.clear()
        try:
            if self._owns_sandbox and self.sandbox_id is not None:
                await self._delete(self.sandbox_id)
        finally:
            await self._close_client()

    async def _delete(self, sandbox_id: str) -> None:
        try:
            response: Final = await self._http().delete(
                f"{self._api_base}/sandboxes/{sandbox_id}", headers=self._headers(), timeout=30.0
            )
        except httpx.HTTPError as exc:
            raise SandboxError(f"could not delete prized sandbox {sandbox_id}: {exc}") from exc
        if response.status_code >= 400 and response.status_code != 404:
            raise SandboxError(f"could not delete prized sandbox {sandbox_id}: {_edge_error(response)}")

    async def _close_client(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()

    async def __aenter__(self) -> PrizedSandbox:
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()


def _under_home(path: str) -> bool:
    return path == PRIZED_HOME or path.startswith(PRIZED_HOME + "/")


class _ErrorBody(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    error: str = ""


def _edge_error(response: httpx.Response) -> str:
    try:
        code: Final = _ErrorBody.model_validate_json(response.content).error
    except ValidationError:
        return f"HTTP {response.status_code}"
    return f"HTTP {response.status_code} {code}".strip()


def _sandbox_doc(response: httpx.Response, action: str) -> _SandboxDoc:
    if response.status_code >= 400:
        raise SandboxError(f"prized sandbox {action} failed: {_edge_error(response)}")
    try:
        return _SandboxEnvelope.model_validate_json(response.content).sandbox
    except ValidationError as exc:
        raise SandboxError(f"prized sandbox {action} returned an unexpected body") from exc


def _exec_answer(response: httpx.Response) -> _ExecAnswer:
    if response.status_code != 200:
        raise SandboxError(f"prized command failed: {_edge_error(response)}")
    try:
        return _ExecAnswer.model_validate_json(response.content)
    except ValidationError as exc:
        raise SandboxError("prized command returned an unexpected body") from exc


def prized(
    template: str = PRIZED_DEFAULT_TEMPLATE,
    *,
    sandbox_id: str | None = None,
    workdir: str = PRIZED_WORKDIR,
    env: Mapping[str, str] | None = None,
    labels: Mapping[str, str] | None = None,
    vcpu: int | None = None,
    memory_mb: int | None = None,
    max_runtime_min: int | None = None,
    api_key: str | None = None,
) -> PrizedSandbox:
    """Sandbox in a new Prized microVM from `template` (or the existing `sandbox_id`), started lazily on first use."""
    return PrizedSandbox(
        template=template,
        sandbox_id=sandbox_id,
        workdir=workdir,
        env=env,
        labels=labels,
        vcpu=vcpu,
        memory_mb=memory_mb,
        max_runtime_min=max_runtime_min,
        api_key=api_key,
    )
