"""Pin v1 streaming: recorded provider events -> sequence of OpenAI chunks.

Anthropic replays raw SSE lines through the real ``ModelResponseIterator``
inside ``CustomStreamWrapper``. The two bedrock routes replay parsed event
payloads through the real decoders' chunk parsers — the binary AWS
event-stream framing upstream of that seam is botocore plumbing, documented
as out of corpus scope in README.md.
"""

import pytest

from ._helpers import FIXTURES_DIR, assert_snapshot, load_json
from ._seams import (
    replay_anthropic_sse,
    replay_bedrock_converse_events,
    replay_bedrock_invoke_events,
)


def _fixture_paths(provider_key: str, suffix: str) -> list:
    return sorted((FIXTURES_DIR / "streams" / provider_key).glob(f"*{suffix}"))


@pytest.mark.parametrize(
    "path", _fixture_paths("anthropic", ".txt"), ids=lambda p: p.stem
)
def test_anthropic_stream(path, snapshot_update: bool) -> None:
    sse_lines = path.read_text().splitlines()
    chunks = replay_anthropic_sse(sse_lines)
    assert_snapshot(f"streams/anthropic/{path.stem}.json", chunks, snapshot_update)


@pytest.mark.parametrize(
    "path", _fixture_paths("bedrock_converse", ".json"), ids=lambda p: p.stem
)
def test_bedrock_converse_stream(path, snapshot_update: bool) -> None:
    chunks = replay_bedrock_converse_events(load_json(path))
    assert_snapshot(
        f"streams/bedrock_converse/{path.stem}.json", chunks, snapshot_update
    )


@pytest.mark.parametrize(
    "path", _fixture_paths("bedrock_invoke", ".json"), ids=lambda p: p.stem
)
def test_bedrock_invoke_stream(path, snapshot_update: bool) -> None:
    chunks = replay_bedrock_invoke_events(load_json(path))
    assert_snapshot(
        f"streams/bedrock_invoke/{path.stem}.json", chunks, snapshot_update
    )
