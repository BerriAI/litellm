"""Differential parity for bedrock streaming, pinned at the parsed-event seam.

v1 side: recorded parsed events through the REAL decoders
(``AWSEventStreamDecoder.converse_chunk_parser`` /
``AmazonAnthropicClaudeStreamDecoder._chunk_parser``) inside
``CustomStreamWrapper`` — the corpus replay method. v2 side:
``engine.stream.fold_events`` with the provider event parser and the
provider's chunk dialect. Chunk lists must match v1 in-process AND the
committed corpus snapshot; stream ids are ambient and normalized.
"""

import copy
import json

import pytest

from litellm.translation.engine.stream import fold_events
from litellm.translation.inbound.openai_chat import parse_request
from litellm.translation.inbound.openai_chat.stream import initial_state
from litellm.translation.providers.bedrock_converse.stream import (
    parse_event as converse_parse_event,
)
from litellm.translation.providers.bedrock_converse.tools import reverse_name_map
from litellm.translation.providers.bedrock_invoke.stream import (
    parse_event as invoke_parse_event,
    reverse_names,
)
from litellm.translation_seam import to_model_response_stream

from . import _bedrock_corpus as corpus

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


def _fixture_ids(provider_key: str) -> list:
    return sorted(
        path.stem
        for path in (corpus.FIXTURES_DIR / "streams" / provider_key).glob("*.json")
    )


def _norm(chunks: list) -> str:
    return json.dumps(
        [{**chunk, "id": "chatcmpl-X"} for chunk in chunks], sort_keys=True, default=str
    )


def _v2_chunks(provider_key: str, events: list) -> list:
    model, _, _ = corpus.resolve(provider_key)
    parsed = parse_request(
        {
            "model": model,
            "max_tokens": 64,
            "tools": copy.deepcopy(_TOOLS),
            "messages": [{"role": "user", "content": "stream"}],
        }
    )
    assert parsed.is_ok(), parsed.error.summary
    if provider_key == "bedrock_converse":
        reverse = reverse_name_map(parsed.ok.tools)
        folded = fold_events(
            events,
            lambda event: converse_parse_event(event, reverse),
            initial_state(model=model, dialect="bedrock_converse"),
        )
    else:
        reverse = dict(reverse_names(parsed.ok))
        folded = fold_events(
            events,
            lambda event: invoke_parse_event(event, reverse),
            initial_state(model=model, dialect="anthropic"),
        )
    assert folded.is_ok(), folded.error.summary
    return [
        to_model_response_stream(chunk, "chatcmpl-X").model_dump()
        for chunk in folded.ok
    ]


@pytest.mark.parametrize(
    "provider_key,fixture_id",
    [(p, f) for p in sorted(corpus.PROVIDERS) for f in _fixture_ids(p)],
)
def test_v2_stream_matches_v1_and_snapshot(
    provider_key: str, fixture_id: str, frozen_ambient
) -> None:
    events = corpus.load_json(
        corpus.FIXTURES_DIR / "streams" / provider_key / f"{fixture_id}.json"
    )
    replay = (
        corpus.replay_v1_converse_events
        if provider_key == "bedrock_converse"
        else corpus.replay_v1_invoke_events
    )
    v1 = replay(copy.deepcopy(events))
    v2 = _v2_chunks(provider_key, events)
    assert _norm(v2) == _norm(v1)
    snapshot = corpus.load_json(
        corpus.SNAPSHOTS_DIR / "streams" / provider_key / f"{fixture_id}.json"
    )
    assert _norm(v2) == _norm(snapshot), (
        f"v2/v1 drifted from the characterization snapshot for {fixture_id}; "
        "regenerate the corpus and ship the diff as its own PR"
    )
