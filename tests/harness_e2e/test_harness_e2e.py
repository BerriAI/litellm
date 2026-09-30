"""End-to-end tests: every harness, real runtime, real LiteLLM AI Gateway."""

from pathlib import Path

import pytest
from pydantic import BaseModel

import litellm
from litellm.harness import (
    CapabilityUnsupported,
    Done,
    FileChange,
    Gateway,
    Harness,
    State,
    Text,
    ToolCall,
)
from litellm import sandbox

from .conftest import harness_params, model_for, requires_gateway

pytestmark = [requires_gateway]

TURN_TIMEOUT = 300


class Answer(BaseModel):
    city: str
    country: str


@pytest.mark.parametrize("harness", harness_params())
def test_run_creates_file_and_reports_cost(
    harness: Harness, gateway: Gateway, workspace: Path
) -> None:
    result = litellm.harness.run(
        harness,
        "Create a file named hello.txt whose entire content is the single word: hi",
        sandbox=sandbox.local(workspace),
        model=model_for(harness),
        gateway=gateway,
        timeout=TURN_TIMEOUT,
    )

    assert result.stop_reason == "done", result.text
    assert (workspace / "hello.txt").read_text().strip().lower() == "hi"
    assert [f.path for f in result.files if f.kind == "created"] == ["hello.txt"]
    assert result.usage.calls >= 1
    assert result.usage.input_tokens > 0
    assert result.cost >= 0


@pytest.mark.parametrize("harness", harness_params())
def test_stream_event_order(
    harness: Harness, gateway: Gateway, workspace: Path
) -> None:
    (workspace / "numbers.txt").write_text("1\n2\n3\n")
    events = list(
        litellm.harness.stream(
            harness,
            "Read numbers.txt and reply with the sum of the numbers in it. Reply with just the number.",
            sandbox=sandbox.local(workspace),
            model=model_for(harness),
            gateway=gateway,
            timeout=TURN_TIMEOUT,
        )
    )

    assert isinstance(events[-1], Done)
    assert sum(isinstance(e, Done) for e in events) == 1
    assert any(isinstance(e, Text) for e in events)
    assert any(isinstance(e, ToolCall) for e in events)
    assert "6" in events[-1].result.text


@pytest.mark.parametrize("harness", harness_params())
def test_structured_output(harness: Harness, gateway: Gateway, workspace: Path) -> None:
    result = litellm.harness.run(
        harness,
        "What is the capital of France? Do not use any tools.",
        sandbox=sandbox.local(workspace),
        model=model_for(harness),
        gateway=gateway,
        output=Answer,
        permissions="read-only",
        timeout=TURN_TIMEOUT,
    )

    assert isinstance(result.output, Answer)
    assert result.output.city.lower() == "paris"


@pytest.mark.parametrize("harness", harness_params())
def test_session_remembers_previous_turn(
    harness: Harness, gateway: Gateway, workspace: Path
) -> None:
    with litellm.harness.session(
        harness,
        sandbox=sandbox.local(workspace),
        model=model_for(harness),
        gateway=gateway,
        timeout=TURN_TIMEOUT,
    ) as s:
        s.run("Remember this code word: PELICAN. Reply with just OK.")
        second = s.run(
            "What code word did I ask you to remember? Reply with just the word."
        )
        assert "pelican" in second.text.lower()
        assert s.cost >= second.cost


@pytest.mark.parametrize("harness", harness_params())
def test_detach_and_resume(harness: Harness, gateway: Gateway, workspace: Path) -> None:
    box = sandbox.local(workspace)
    s = litellm.harness.session(
        harness,
        sandbox=box,
        model=model_for(harness),
        gateway=gateway,
        timeout=TURN_TIMEOUT,
    )
    s.run("Remember this number: 4817. Reply with just OK.")
    raw = s.detach().dumps()

    resumed = litellm.harness.resume(State.loads(raw), sandbox=box, gateway=gateway)
    with resumed:
        r = resumed.run(
            "What number did I ask you to remember? Reply with just the number."
        )
    assert "4817" in r.text


@pytest.mark.parametrize("harness", harness_params())
def test_read_only_blocks_writes(
    harness: Harness, gateway: Gateway, workspace: Path
) -> None:
    result = litellm.harness.run(
        harness,
        "Create a file named blocked.txt containing x. If you cannot, just say you cannot.",
        sandbox=sandbox.local(workspace),
        model=model_for(harness),
        gateway=gateway,
        permissions="read-only",
        timeout=TURN_TIMEOUT,
    )

    assert not (workspace / "blocked.txt").exists()
    assert not [f for f in result.files if isinstance(f, FileChange)]


def test_string_harness_rejected(workspace: Path) -> None:
    with pytest.raises(TypeError, match="Harness.CODEX"):
        litellm.harness.run("codex", "hi", sandbox=sandbox.local(workspace))  # type: ignore[arg-type]


def test_capability_checked_before_start(gateway: Gateway, workspace: Path) -> None:
    with pytest.raises(CapabilityUnsupported):
        litellm.harness.run(
            Harness.CODEX,
            "hi",
            sandbox=sandbox.local(workspace),
            gateway=gateway,
            disable_tools=["bash"],
        )
