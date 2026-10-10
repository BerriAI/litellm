import os
import subprocess
import sys
import textwrap
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

import pytest
from integration._support import device_login as dl
from integration._support import responses_vendor as rv
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(300)

Provider: TypeAlias = Literal["chatgpt", "copilot"]

_MODELS: Final[Mapping[Provider, str]] = MappingProxyType(
    {"chatgpt": "chatgpt/gpt-5.6-terra", "copilot": "github_copilot/gpt-5.4"}
)
_REFUSALS: Final[Mapping[Provider, str]] = MappingProxyType(
    {"chatgpt": dl.CHATGPT_REFUSAL, "copilot": dl.COPILOT_REFUSAL}
)
_USER_CODES: Final[Mapping[Provider, str]] = MappingProxyType(
    {"chatgpt": dl.CHATGPT_USER_CODE, "copilot": dl.COPILOT_USER_CODE}
)
_AUTHENTICATORS: Final[Mapping[Provider, str]] = MappingProxyType(
    {"chatgpt": "litellm.llms.chatgpt.authenticator", "copilot": "litellm.llms.github_copilot.authenticator"}
)
_SCRIPT_SECONDS: Final = 60

_SCRIPT: Final = textwrap.dedent(
    """
    import asyncio, json, sys, threading
    import litellm
    from litellm import Router

    variant, model, marker, authenticator_module = sys.argv[1:5]
    messages = [{"role": "user", "content": f"Reply to marker-{marker}"}]

    def sync_completion():
        return litellm.completion(model=model, messages=messages, num_retries=0)

    def run():
        if variant == "completion":
            return sync_completion()
        if variant == "responses":
            return litellm.responses(model=model, input=f"Reply to marker-{marker}", num_retries=0)
        if variant == "acompletion":
            return asyncio.run(litellm.acompletion(model=model, messages=messages, num_retries=0))
        if variant == "aresponses":
            return asyncio.run(litellm.aresponses(model=model, input=f"Reply to marker-{marker}", num_retries=0))
        if variant == "worker-thread":
            outcome = {}
            def target():
                try:
                    outcome["value"] = sync_completion()
                except Exception as error:
                    outcome["error"] = error
            thread = threading.Thread(target=target)
            thread.start()
            thread.join()
            if "error" in outcome:
                raise outcome["error"]
            return outcome["value"]
        if variant == "batch-completion":
            (only,) = litellm.batch_completion(model=model, messages=[messages], num_retries=0)
            if isinstance(only, Exception):
                raise only
            return only
        if variant == "sync-inside-loop":
            async def main():
                return sync_completion()
            return asyncio.run(main())
        if variant == "one-liner-inside-loop":
            module = __import__(authenticator_module, fromlist=["Authenticator"])
            async def main():
                return module.Authenticator().get_access_token()
            return asyncio.run(main())
        if variant == "router-built-in-async":
            async def main():
                router = Router(model_list=[{"model_name": "login", "litellm_params": {"model": model}}])
                return await router.acompletion(model="login", messages=messages, num_retries=0)
            return asyncio.run(main())
        if variant == "wildcard-router-in-loop":
            router = Router(model_list=[{"model_name": "*", "litellm_params": {"model": "*"}}])
            async def main():
                return await router.acompletion(model=model, messages=messages, num_retries=0)
            return asyncio.run(main())
        raise SystemExit(f"unknown variant {variant}")

    def render(result):
        if hasattr(result, "__next__"):
            return "".join(str(chunk) for chunk in result)
        return str(result)

    try:
        print(json.dumps({"ok": True, "text": render(run())}), flush=True)
    except Exception as error:
        print(json.dumps({"ok": False, "error": f"{type(error).__name__}: {error}"}), flush=True)
    """
)


@dataclass(frozen=True, slots=True)
class _Outcome:
    returncode: int
    stdout: str
    stderr: str

    def verdict(self) -> Mapping[str, JsonValue]:
        lines: Final = [line for line in self.stdout.splitlines() if line.startswith("{")]
        assert lines, (self.stdout, self.stderr)
        return rv.JSON_OBJECT.validate_json(lines[-1])


@dataclass(frozen=True, slots=True)
class _Bench:
    peers: dl.Peers
    chatgpt_dir: Path
    copilot_dir: Path

    def run(self, variant: str, provider: Provider, marker: str) -> _Outcome:
        completed: Final = subprocess.run(
            [sys.executable, "-P", "-c", _SCRIPT, variant, _MODELS[provider], marker, _AUTHENTICATORS[provider]],
            env={
                **os.environ,
                **self.peers.environment(),
                "CHATGPT_TOKEN_DIR": str(self.chatgpt_dir),
                "GITHUB_COPILOT_TOKEN_DIR": str(self.copilot_dir),
                "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            },
            capture_output=True,
            text=True,
            timeout=_SCRIPT_SECONDS,
            check=False,
        )
        return _Outcome(completed.returncode, completed.stdout, completed.stderr)

    def token_files(self) -> tuple[str, ...]:
        return tuple(sorted(path.name for path in (*self.chatgpt_dir.iterdir(), *self.copilot_dir.iterdir())))


@pytest.fixture(scope="module")
def bench(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Bench]:
    directory: Final = tmp_path_factory.mktemp("device-login-sdk")
    chatgpt_dir: Final = directory / "chatgpt"
    copilot_dir: Final = directory / "copilot"
    chatgpt_dir.mkdir()
    copilot_dir.mkdir()
    with dl.device_login_peers(directory) as peers:
        yield _Bench(peers, chatgpt_dir, copilot_dir)


@pytest.fixture
def clean(bench: _Bench) -> Iterator[_Bench]:
    bench.peers.reset()
    dl.clear_tokens(bench.chatgpt_dir, bench.copilot_dir)
    yield bench
    bench.peers.reset()
    dl.clear_tokens(bench.chatgpt_dir, bench.copilot_dir)


_FIRST_LOGIN_BEARERS: Final[Mapping[Provider, str]] = MappingProxyType(
    {"chatgpt": dl.CHATGPT_FIRST_LOGIN, "copilot": dl.COPILOT_MINTED_KEY}
)


@pytest.mark.parametrize("variant", ("completion", "responses"))
@pytest.mark.parametrize("provider", ("chatgpt", "copilot"))
def test_sync_call_on_the_main_thread_keeps_the_interactive_first_login(
    clean: _Bench, provider: Provider, variant: str
) -> None:
    clean.peers.switches.grant.set()
    marker: Final = uuid.uuid4().hex
    outcome: Final = clean.run(variant, provider, marker)
    verdict: Final = outcome.verdict()
    assert verdict["ok"] is True, (verdict, outcome.stderr[-3000:])
    assert marker in str(verdict["text"]), verdict
    assert _USER_CODES[provider] in outcome.stdout, outcome.stdout
    assert f"{dl.CHATGPT_AUTH_HOST if provider == 'chatgpt' else dl.GITHUB_HOST}:443" in clean.peers.auth_connections()
    forwarded: Final = [request for request in clean.peers.api.drain() if request.method == "POST"]
    assert {dl.bearer(request) for request in forwarded} == {_FIRST_LOGIN_BEARERS[provider]}, [
        (request.method, request.target) for request in forwarded
    ]
    expected_files: Final = ("auth.json",) if provider == "chatgpt" else ("access-token", "api-key.json")
    assert clean.token_files() == expected_files


_IN_LOOP_VARIANTS: Final = (
    "acompletion",
    "aresponses",
    "worker-thread",
    "batch-completion",
    "sync-inside-loop",
    "one-liner-inside-loop",
    "router-built-in-async",
    "wildcard-router-in-loop",
)


@pytest.mark.parametrize("variant", _IN_LOOP_VARIANTS)
@pytest.mark.parametrize("provider", ("chatgpt", "copilot"))
def test_a_call_inside_a_loop_or_worker_thread_is_refused_before_any_auth_host_call(
    clean: _Bench, provider: Provider, variant: str
) -> None:
    clean.peers.switches.grant.set()
    outcome: Final = clean.run(variant, provider, uuid.uuid4().hex)
    verdict: Final = outcome.verdict()
    assert verdict["ok"] is False, (verdict, outcome.stderr[-3000:])
    assert _REFUSALS[provider] in str(verdict["error"]), verdict
    assert _USER_CODES[provider] not in outcome.stdout, outcome.stdout
    assert clean.peers.auth_connections() == ()
    assert clean.peers.api.drain() == ()
    assert clean.token_files() == ()


@pytest.mark.parametrize("provider", ("chatgpt", "copilot"))
def test_sync_call_on_the_main_thread_fails_without_hanging_when_the_auth_host_denies(
    clean: _Bench, provider: Provider
) -> None:
    outcome: Final = clean.run("completion", provider, uuid.uuid4().hex)
    verdict: Final = outcome.verdict()
    assert verdict["ok"] is False, (verdict, outcome.stderr[-3000:])
    assert f"{dl.CHATGPT_AUTH_HOST if provider == 'chatgpt' else dl.GITHUB_HOST}:443" in clean.peers.auth_connections()
    assert clean.peers.api.drain() == ()
    assert clean.token_files() == ()
