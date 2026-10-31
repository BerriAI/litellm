import asyncio
import json
import shlex
from collections.abc import Callable
from typing import Final

import httpx
import pytest

from litellm import sandbox
from litellm.harness.errors import CapabilityUnsupported, SandboxError
from litellm.harness.sandbox import PrizedSandbox, Sandbox

API: Final = "https://api.prized.dev/api/v1"
EDGE: Final = "https://term.prized.dev/v1/sandbox/sbx_1"
ExecReply = Callable[[dict[str, object]], httpx.Response]


class FakePrized:
    """In-memory Prized API: one sandbox, a file store, and scripted command answers."""

    def __init__(self, exec_reply: ExecReply | None = None, *, unready_answers: int = 0) -> None:
        self.creates: list[dict[str, object]] = []
        self.commands: list[dict[str, object]] = []
        self.files: dict[str, bytes] = {}
        self.deleted: list[str] = []
        self.unready_answers = unready_answers
        self.exec_reply = exec_reply or (lambda body: httpx.Response(200, json={"stdout": "", "exitCode": 0}))

    def _sandbox(self, state: str = "running") -> dict[str, object]:
        return {"sandbox": {"id": "sbx_1", "state": state, "toolboxBase": EDGE}}

    def handler(self, request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer dcp_test"
        path = request.url.path
        if request.method == "POST" and path == "/api/v1/sandboxes":
            self.creates.append(json.loads(request.content))
            return httpx.Response(201, json=self._sandbox())
        if request.method == "GET" and path == "/api/v1/sandboxes/sbx_1":
            return httpx.Response(200, json=self._sandbox())
        if request.method == "DELETE" and path == "/api/v1/sandboxes/sbx_1":
            self.deleted.append("sbx_1")
            return httpx.Response(202, json=self._sandbox("deleting"))
        if path.endswith("/process/execute"):
            body = json.loads(request.content)
            if body["command"] == "true" and self.unready_answers:
                self.unready_answers -= 1
                return httpx.Response(502, json={"error": "sandbox_unreachable"})
            self.commands.append(body)
            return self.exec_reply(body)
        if path.endswith("/files/upload"):
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


def user_commands(fake: FakePrized) -> list[str]:
    return [
        str(c["command"]) for c in fake.commands if c["command"] != "true" and "mkdir -p --" not in str(c["command"])
    ]


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
