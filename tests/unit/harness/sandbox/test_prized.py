import asyncio
import importlib
import json
import re
import shlex
from collections.abc import Awaitable, Callable
from typing import Final

import httpx
import pytest

from litellm import sandbox
from litellm.harness.errors import CapabilityUnsupported, SandboxError
from litellm.harness.sandbox import PrizedSandbox, Sandbox
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.types.llms.custom_http import httpxSpecialProvider

API: Final = "https://api.prized.dev/api/v1"
EDGE: Final = "https://term.prized.dev/v1/sandbox/sbx_1"
ExecReply = Callable[[dict[str, object]], httpx.Response | Awaitable[httpx.Response]]
# The module, not the `prized()` factory that litellm.harness.sandbox re-exports under the same name.
prized_module: Final = importlib.import_module("litellm.harness.sandbox.prized")


class FakePrized:
    """In-memory Prized API: one sandbox, a file store, and scripted command answers."""

    def __init__(
        self,
        exec_reply: ExecReply | None = None,
        *,
        unready_answers: int = 0,
        states: tuple[str, ...] = ("running",),
        toolbox: str | None = EDGE,
        delete_answer: httpx.Response | httpx.HTTPError | None = None,
        read_only: bool = False,
    ) -> None:
        self.creates: list[dict[str, object]] = []
        self.commands: list[dict[str, object]] = []
        self.files: dict[str, bytes] = {}
        self.deleted: list[str] = []
        self.unready_answers = unready_answers
        self.exec_reply = exec_reply or (lambda body: httpx.Response(200, json={"stdout": "", "exitCode": 0}))
        self.states = list(states)
        self.toolbox = toolbox
        self.delete_answer = delete_answer
        self.read_only = read_only

    def _sandbox(self, state: str | None = None) -> dict[str, object]:
        """The next scripted state; the last one repeats."""
        current = state or (self.states.pop(0) if len(self.states) > 1 else self.states[0])
        return {"sandbox": {"id": "sbx_1", "state": current, "toolboxBase": self.toolbox, "errorReason": "boom"}}

    async def handler(self, request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer dcp_test"
        path = request.url.path
        if request.method == "POST" and path == "/api/v1/sandboxes":
            self.creates.append(json.loads(request.content))
            return httpx.Response(201, json=self._sandbox())
        if request.method == "GET" and path == "/api/v1/sandboxes/sbx_1":
            return httpx.Response(200, json=self._sandbox())
        if request.method == "DELETE" and path == "/api/v1/sandboxes/sbx_1":
            self.deleted.append("sbx_1")
            if isinstance(self.delete_answer, httpx.HTTPError):
                raise self.delete_answer
            return self.delete_answer or httpx.Response(202, json=self._sandbox("deleting"))
        if path.endswith("/process/execute"):
            body = json.loads(request.content)
            if body["command"] == "true" and self.unready_answers:
                self.unready_answers -= 1
                return httpx.Response(502, json={"error": "sandbox_unreachable"})
            self.commands.append(body)
            reply = self.exec_reply(body)
            return reply if isinstance(reply, httpx.Response) else await reply
        if path.endswith("/files/upload"):
            if self.read_only:
                return httpx.Response(403, json={"error": "forbidden"})
            self.files[request.url.params["path"]] = request.content
            return httpx.Response(200, json={"path": request.url.params["path"], "size": len(request.content)})
        if path.endswith("/files/download"):
            data = self.files.get(request.url.params["path"])
            return (
                httpx.Response(404, json={"error": "not_found"}) if data is None else httpx.Response(200, content=data)
            )
        return httpx.Response(404, json={"error": "not_found"})


def make_box(fake: FakePrized, **kwargs: object) -> PrizedSandbox:
    client = httpx.AsyncClient(transport=httpx.MockTransport(fake.handler))
    return PrizedSandbox(api_key="dcp_test", client=client, **kwargs)  # pyright: ignore[reportArgumentType]  # test kwargs


def box_answering(response: httpx.Response, **kwargs: object) -> PrizedSandbox:
    """A sandbox whose API answers every request with `response`."""
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: response))
    return PrizedSandbox(api_key="dcp_test", client=client, **kwargs)  # pyright: ignore[reportArgumentType]  # test kwargs


def user_commands(fake: FakePrized) -> list[str]:
    return [
        str(c["command"]) for c in fake.commands if c["command"] != "true" and "mkdir -p --" not in str(c["command"])
    ]


def is_setup(body: dict[str, object]) -> bool:
    """The readiness probe and workdir mkdir that start() sends before any user command."""
    command = str(body["command"])
    return command == "true" or command.startswith("mkdir -p --")


class Hold:
    """Exec reply that answers setup and stop commands at once and holds each user command until released."""

    def __init__(self) -> None:
        self.arrived = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, body: dict[str, object]) -> httpx.Response:
        if not is_setup(body) and "kill -TERM" not in str(body["command"]):
            self.arrived.set()
            await self.release.wait()
        return httpx.Response(200, json={"exitCode": 0})


@pytest.fixture
def no_waits(monkeypatch):
    """Poll without sleeping and give up on the first missed deadline."""
    monkeypatch.setattr(prized_module, "_POLL_SECONDS", 0)
    monkeypatch.setattr(prized_module, "_START_TIMEOUT_SECONDS", -1.0)
    monkeypatch.setattr(prized_module, "_READY_TIMEOUT_SECONDS", -1.0)


def test_prized_satisfies_protocol_and_is_exported():
    box = sandbox.prized()
    assert isinstance(box, Sandbox)
    assert box.workdir == "/home/prized/work"


@pytest.mark.parametrize("workdir", ["/workspace", "work", "/home/prized/../etc"])
def test_workdir_must_be_under_the_sandbox_home(workdir):
    with pytest.raises(SandboxError):
        sandbox.prized(workdir=workdir)


async def test_concurrent_first_use_creates_one_sandbox_and_runs_argv_literally():
    fake = FakePrized(lambda body: httpx.Response(200, json={"stdout": "out", "stderr": "err", "exitCode": 7}))
    box = make_box(fake, env={"A": "1"})
    payload = "a b; $(rm -rf /) 'q'"
    results = await asyncio.gather(*(box.run(["printf", "%s", payload], env={"B": "2"}) for _ in range(3)))
    assert len(fake.creates) == 1
    assert {(r.exit_code, r.stdout, r.stderr) for r in results} == {(7, "out", "err")}
    command = user_commands(fake)[0]
    assert shlex.join(["printf", "%s", payload]) in command
    assert fake.commands[-1]["env"] == {"A": "1", "B": "2"}
    assert fake.commands[-1]["cwd"] == "/home/prized/work"


async def test_first_command_waits_out_an_edge_that_is_not_ready_yet():
    fake = FakePrized(unready_answers=1)
    box = make_box(fake)
    assert (await box.run(["true"])).exit_code == 0


async def test_a_command_past_its_timeout_raises():
    fake = FakePrized(lambda body: httpx.Response(200, json={"exitCode": -1, "timedOut": True}))
    box = make_box(fake)
    with pytest.raises(SandboxError, match="timed out"):
        await box.run(["sleep", "10"], timeout=1)
    assert fake.commands[-1]["timeout"] == 1


async def test_files_round_trip_through_the_file_api_with_absolute_paths():
    fake = FakePrized()
    box = make_box(fake)
    await box.write("src/a.py", b"\x00data\xff")
    assert fake.files == {"/home/prized/work/src/a.py": b"\x00data\xff"}
    assert await box.read("/home/prized/work/src/a.py") == b"\x00data\xff"
    with pytest.raises(SandboxError):
        await box.read("missing.txt")


async def test_snapshot_maps_relative_paths_to_hashes():
    digest = "a" * 64
    fake = FakePrized(lambda body: httpx.Response(200, json={"stdout": f"{digest}  ./src/a.py\n", "exitCode": 0}))
    box = make_box(fake)
    assert dict(await box.snapshot()) == {"src/a.py": digest}


async def test_exec_delivers_output_on_the_readers():
    fake = FakePrized(lambda body: httpx.Response(200, json={"stdout": "hello\n", "stderr": "", "exitCode": 3}))
    box = make_box(fake)
    process = await box.exec(["echo", "hello"])
    assert await process.wait() == 3
    assert await process.stdout.read() == b"hello\n"


async def test_close_deletes_a_created_sandbox_but_not_an_attached_one():
    created = FakePrized()
    box = make_box(created)
    await box.run(["true"])
    await box.close()
    assert created.deleted == ["sbx_1"]

    attached = FakePrized()
    other = make_box(attached, sandbox_id="sbx_1")
    await other.run(["true"])
    await other.close()
    assert attached.creates == [] and attached.deleted == []


def test_cli_harnesses_cannot_reach_this_machine():
    with pytest.raises(CapabilityUnsupported):
        sandbox.prized().host_url(4000)


async def test_without_an_injected_client_it_uses_the_shared_client_and_leaves_it_open():
    fake = FakePrized()
    shared = get_async_httpx_client(llm_provider=httpxSpecialProvider.Sandbox)
    created = shared.client
    shared.client = httpx.AsyncClient(transport=httpx.MockTransport(fake.handler))
    box = PrizedSandbox(api_key="dcp_test")
    assert (await box.run(["true"])).exit_code == 0
    await box.close()
    assert len(fake.creates) == 1 and fake.deleted == ["sbx_1"]
    assert not shared.client.is_closed
    await shared.client.aclose()
    await created.aclose()


async def test_the_api_key_comes_from_prized_api_key_and_a_missing_one_fails_before_any_request(monkeypatch):
    monkeypatch.delenv("PRIZED_API_KEY", raising=False)
    fake = FakePrized()
    keyless = PrizedSandbox(client=httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)))
    with pytest.raises(SandboxError, match="PRIZED_API_KEY"):
        await keyless.start()
    assert fake.creates == []

    monkeypatch.setenv("PRIZED_API_KEY", "dcp_test")
    from_env = PrizedSandbox(client=httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)))
    assert await from_env.start() == "sbx_1"


async def test_create_sends_the_template_labels_env_and_resources():
    fake = FakePrized()
    box = make_box(
        fake, template="py-3.12", env={"A": "1"}, labels={"team": "ml"}, vcpu=4, memory_mb=8192, max_runtime_min=30
    )
    assert repr(box) == "PrizedSandbox('py-3.12', workdir='/home/prized/work')"
    await box.start()
    assert repr(box) == "PrizedSandbox('sbx_1', workdir='/home/prized/work')"
    assert fake.creates == [
        {
            "template": "py-3.12",
            "labels": {"created-by": "litellm-harness", "team": "ml"},
            "envVars": {"A": "1"},
            "resources": {"vcpu": 4, "memoryMb": 8192},
            "maxRuntimeMin": 30,
        }
    ]
    assert fake.commands[-1]["command"] == "mkdir -p -- /home/prized/work"


async def test_start_polls_until_a_new_sandbox_is_running(monkeypatch):
    monkeypatch.setattr(prized_module, "_POLL_SECONDS", 0)
    fake = FakePrized(states=("creating", "starting", "running"))
    box = make_box(fake)
    assert await box.start() == "sbx_1"
    assert fake.states == ["running"]


@pytest.mark.parametrize("state", ["error", "deleted"])
async def test_start_fails_on_a_sandbox_that_will_never_run(state):
    box = make_box(FakePrized(states=(state,)))
    with pytest.raises(SandboxError, match=f"is {state}: boom"):
        await box.start()


async def test_start_fails_when_the_sandbox_has_no_command_endpoint():
    box = make_box(FakePrized(states=("paused",), toolbox=None))
    with pytest.raises(SandboxError, match="no command endpoint"):
        await box.start()


async def test_start_gives_up_on_a_sandbox_that_never_runs(no_waits):
    box = make_box(FakePrized(states=("creating",)))
    with pytest.raises(SandboxError, match="did not start"):
        await box.start()


async def test_start_gives_up_on_an_edge_that_never_answers(no_waits):
    box = make_box(FakePrized(unready_answers=1_000))
    with pytest.raises(SandboxError, match=re.escape("did not answer commands (502)")):
        await box.start()


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(401, json={"error": "unauthorized"}), "create failed: HTTP 401 unauthorized"),
        (httpx.Response(503, text="<html>bad gateway</html>"), "create failed: HTTP 503"),
        (httpx.Response(201, json={"sandboxes": []}), "create returned an unexpected body"),
    ],
)
async def test_create_errors_name_the_edge_error(response, message):
    with pytest.raises(SandboxError, match=message):
        await box_answering(response).start()


async def test_attaching_to_a_missing_sandbox_names_it():
    box = box_answering(httpx.Response(404, json={"error": "not_found"}), sandbox_id="sbx_9")
    with pytest.raises(SandboxError, match="get sbx_9 failed: HTTP 404 not_found"):
        await box.start()


async def test_run_and_exec_reject_an_empty_command_and_a_timeout_past_the_limit():
    box = make_box(FakePrized())
    with pytest.raises(SandboxError, match=re.escape("run() needs a non-empty command")):
        await box.run([])
    with pytest.raises(SandboxError, match=re.escape("exec() needs a non-empty command")):
        await box.exec([])
    with pytest.raises(SandboxError, match="at most 3600s"):
        await box.run(["sleep", "4000"], timeout=4000)


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(500, json={"error": "internal"}), "command failed: HTTP 500 internal"),
        (httpx.Response(200, json={"stdout": "no exit code"}), "command returned an unexpected body"),
    ],
)
async def test_a_failed_command_raises_and_tries_to_stop_its_process(response, message):
    fake = FakePrized(lambda body: httpx.Response(200, json={"exitCode": 0}) if is_setup(body) else response)
    box = make_box(fake)
    with pytest.raises(SandboxError, match=message):
        await box.run(["make", "test"])
    assert "kill -TERM" in user_commands(fake)[-1]


async def test_close_kills_a_running_command_and_deletes_the_sandbox():
    hold = Hold()
    fake = FakePrized(hold)
    box = make_box(fake)
    process = await box.exec(["sleep", "600"])
    await asyncio.wait_for(hold.arrived.wait(), 5)
    await box.close()
    assert await process.wait() == -9
    assert await process.stdout.read() == b""
    run_line, stop_line = user_commands(fake)
    pid_file = re.search(r"/tmp/litellm-harness-runs/[0-9a-f]+\.pid", run_line)
    assert pid_file is not None and pid_file.group(0) in stop_line
    assert fake.deleted == ["sbx_1"]


async def test_cancelling_a_wait_leaves_the_command_running():
    hold = Hold()
    fake = FakePrized(hold)
    box = make_box(fake)
    process = await box.exec(["sleep", "600"])
    waiter = asyncio.ensure_future(process.wait())
    await asyncio.wait_for(hold.arrived.wait(), 5)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert not process.done.done()
    hold.release.set()
    assert await process.wait() == 0
    assert len(user_commands(fake)) == 1


async def test_a_closed_sandbox_cannot_start_again():
    fake = FakePrized()
    async with make_box(fake) as box:
        await box.run(["true"])
    await box.close()
    assert fake.deleted == ["sbx_1"]
    with pytest.raises(SandboxError, match="closed"):
        await box.run(["true"])


@pytest.mark.parametrize(
    ("answer", "error"),
    [
        (httpx.Response(404, json={"error": "not_found"}), None),
        (httpx.Response(500, json={"error": "internal"}), "HTTP 500 internal"),
        (httpx.ConnectError("edge down"), "edge down"),
    ],
)
async def test_close_ignores_a_sandbox_already_gone_and_reports_other_delete_failures(answer, error):
    fake = FakePrized(delete_answer=answer)
    box = make_box(fake)
    await box.start()
    if error is None:
        await box.close()
    else:
        with pytest.raises(SandboxError, match=f"could not delete prized sandbox sbx_1: {error}"):
            await box.close()
    assert fake.deleted == ["sbx_1"]


async def test_a_rejected_write_names_the_edge_error():
    box = make_box(FakePrized(read_only=True))
    with pytest.raises(SandboxError, match=re.escape("could not write /home/prized/work/a.txt: HTTP 403 forbidden")):
        await box.write("a.txt", b"data")


@pytest.mark.parametrize(
    ("stdout", "exit_code", "found"),
    [("/usr/bin/git\n", 0, "/usr/bin/git"), ("", 1, None)],
)
async def test_which_returns_the_path_only_when_found(stdout, exit_code, found):
    fake = FakePrized(lambda body: httpx.Response(200, json={"stdout": stdout, "exitCode": exit_code}))
    box = make_box(fake)
    assert await box.which("git") == found
    assert shlex.join(["sh", "-lc", 'command -v "$1"', "sh", "git"]) in user_commands(fake)[-1]


async def test_tempdir_returns_a_new_directory_under_home_and_raises_when_mktemp_fails():
    fake = FakePrized(
        lambda body: httpx.Response(200, json={"stdout": "/home/prized/litellm-harness-x1\n", "exitCode": 0})
    )
    made = make_box(fake)
    assert await made.tempdir() == "/home/prized/litellm-harness-x1"
    assert "mktemp -d -p /home/prized litellm-harness-XXXXXX" in user_commands(fake)[-1]
    refused = make_box(
        FakePrized(
            lambda body: (
                httpx.Response(200, json={"exitCode": 0})
                if is_setup(body)
                else httpx.Response(200, json={"stderr": "no space\n", "exitCode": 1})
            )
        )
    )
    with pytest.raises(SandboxError, match="mktemp -d failed: no space"):
        await refused.tempdir()


async def test_a_failed_snapshot_raises():
    box = make_box(
        FakePrized(
            lambda body: (
                httpx.Response(200, json={"exitCode": 0})
                if is_setup(body)
                else httpx.Response(200, json={"stderr": "sha256sum: not found\n", "exitCode": 127})
            )
        )
    )
    with pytest.raises(SandboxError, match="snapshot failed: sha256sum: not found"):
        await box.snapshot()
