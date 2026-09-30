"""End-to-end: every harness, real runtime, real LiteLLM AI Gateway via litellm_proxy/."""

from pathlib import Path

import pytest
from pydantic import BaseModel

import litellm
from litellm import Harness, sandbox
from litellm.harness import (
    CapabilityUnsupported,
    Done,
    FileChange,
    State,
    Text,
    ToolCall,
)

from .conftest import harness_params, model_for, requires_gateway

pytestmark = [requires_gateway]

TURN_TIMEOUT = 300


class Answer(BaseModel):
    city: str
    country: str


@pytest.mark.parametrize("harness", harness_params())
def test_agent_creates_file_and_reports_cost(harness: Harness, workspace: Path) -> None:
    result = litellm.agent(
        harness,
        "Create a file named hello.txt whose entire content is the single word: hi",
        sandbox=sandbox.local(workspace),
        model=model_for(harness),
        timeout=TURN_TIMEOUT,
    )

    assert result.stop_reason == "done", result.text
    assert (workspace / "hello.txt").read_text().strip().lower() == "hi"
    assert [f.path for f in result.files if f.kind == "created"] == ["hello.txt"]
    assert result.usage.calls >= 1
    assert result.usage.input_tokens > 0
    assert result.cost >= 0


@pytest.mark.parametrize("harness", harness_params())
def test_agent_stream_event_order(harness: Harness, workspace: Path) -> None:
    (workspace / "secret.txt").write_text("The secret word is ZEBRA.\n")
    events = list(
        litellm.agent(
            harness,
            "Read secret.txt and reply with just the secret word in it.",
            sandbox=sandbox.local(workspace),
            model=model_for(harness),
            timeout=TURN_TIMEOUT,
            stream=True,
        )
    )

    assert isinstance(events[-1], Done)
    assert sum(isinstance(e, Done) for e in events) == 1
    assert any(isinstance(e, Text) for e in events)
    assert any(isinstance(e, ToolCall) for e in events)
    assert "zebra" in events[-1].result.text.lower()


@pytest.mark.parametrize("harness", harness_params())
def test_agent_structured_output(harness: Harness, workspace: Path) -> None:
    result = litellm.agent(
        harness,
        "What is the capital of France? Do not use any tools.",
        sandbox=sandbox.local(workspace),
        model=model_for(harness),
        output=Answer,
        permissions="read-only",
        timeout=TURN_TIMEOUT,
    )

    assert isinstance(result.output, Answer)
    assert result.output.city.lower() == "paris"


@pytest.mark.parametrize("harness", harness_params())
def test_agent_session_remembers_previous_turn(
    harness: Harness, workspace: Path
) -> None:
    with litellm.agent_session(
        harness,
        sandbox=sandbox.local(workspace),
        model=model_for(harness),
        timeout=TURN_TIMEOUT,
    ) as s:
        s.run("Remember this code word: PELICAN. Reply with just OK.")
        second = s.run(
            "What code word did I ask you to remember? Reply with just the word."
        )
        assert "pelican" in second.text.lower()
        assert s.cost >= second.cost


@pytest.mark.parametrize("harness", harness_params())
def test_agent_detach_and_resume(harness: Harness, workspace: Path) -> None:
    box = sandbox.local(workspace)
    s = litellm.agent_session(
        harness, sandbox=box, model=model_for(harness), timeout=TURN_TIMEOUT
    )
    s.run("Remember this number: 4817. Reply with just OK.")
    raw = s.detach().dumps()

    with litellm.agent_resume(
        State.loads(raw), sandbox=box, model=model_for(harness)
    ) as resumed:
        r = resumed.run(
            "What number did I ask you to remember? Reply with just the number."
        )
    assert "4817" in r.text


@pytest.mark.parametrize("harness", harness_params())
def test_agent_read_only_blocks_writes(harness: Harness, workspace: Path) -> None:
    result = litellm.agent(
        harness,
        "Create a file named blocked.txt containing x. If you cannot, just say you cannot.",
        sandbox=sandbox.local(workspace),
        model=model_for(harness),
        permissions="read-only",
        timeout=TURN_TIMEOUT,
    )

    assert not (workspace / "blocked.txt").exists()
    assert not [f for f in result.files if isinstance(f, FileChange)]


def test_string_harness_rejected(workspace: Path) -> None:
    with pytest.raises(TypeError, match=r"Harness\.CODEX"):
        litellm.agent("codex", "hi", sandbox=sandbox.local(workspace))  # type: ignore[arg-type]


def test_capability_checked_before_start(workspace: Path) -> None:
    with pytest.raises(CapabilityUnsupported):
        litellm.agent(
            Harness.CODEX,
            "hi",
            sandbox=sandbox.local(workspace),
            model=model_for(Harness.CODEX),
            disable_tools=["bash"],
        )
