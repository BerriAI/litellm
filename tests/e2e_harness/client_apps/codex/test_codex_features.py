from __future__ import annotations

from pathlib import Path
from typing import Final

from client_apps.codex.events import Transcript, parse_transcript
from client_apps.codex.checks import named_by_eye

TRANSCRIPTS: Final = Path(__file__).resolve().parent / "transcripts"


def _transcript(name: str) -> Transcript:
    return parse_transcript((TRANSCRIPTS / name).read_text())


def test_vision_passes_a_color_named_from_the_image_alone() -> None:
    assert named_by_eye("red")(_transcript("vision_by_eye.jsonl")) is None


def test_vision_fails_a_color_read_from_the_png_through_the_shell() -> None:
    error: Final = named_by_eye("green")(_transcript("vision_read_by_shell.jsonl"))

    assert error is not None
    assert "through the shell" in error


def test_vision_fails_a_wrong_color_named_from_the_image() -> None:
    error: Final = named_by_eye("blue")(_transcript("vision_by_eye.jsonl"))

    assert error is not None
    assert "lacks 'blue'" in error
