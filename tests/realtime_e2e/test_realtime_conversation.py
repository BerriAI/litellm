"""Layer 1: normal text conversation through the proxy, per provider.

Proves the proxy normalizes each provider's realtime stream into the OpenAI GA
event schema: session lifecycle, the canonical response event sequence, that
deltas reconstruct the final transcript, and that usage is reported.
"""

from __future__ import annotations

import pytest

from .conftest import skip_if_creds_missing
from .providers import PROVIDER_IDS, PROVIDERS, RealtimeProvider
from .raw_ws_client import (
    connect_realtime,
    is_type,
    last_of_type,
    of_type,
    response_transcript,
    user_text_event,
)

pytestmark = [pytest.mark.realtime_e2e, pytest.mark.asyncio]


@pytest.mark.parametrize("provider", PROVIDERS, ids=PROVIDER_IDS)
async def test_text_conversation(
    provider: RealtimeProvider, proxy_ws_url: str, proxy_api_key: str
) -> None:
    skip_if_creds_missing(provider)

    async with connect_realtime(
        proxy_ws_url=proxy_ws_url, api_key=proxy_api_key, model=provider.model
    ) as conn:
        created = await conn.collect_until(is_type("session.created"), timeout=20)
        assert created[-1].type == "session.created"

        await conn.send_event(
            {
                "type": "session.update",
                "session": {
                    "modalities": ["text"],
                    "instructions": "You are a terse assistant. Reply in one short sentence.",
                },
            }
        )
        updated = await conn.collect_until(is_type("session.updated"), timeout=20)
        session = updated[-1].raw["session"]
        assert "text" in session.get("modalities", [])

        await conn.send_event(user_text_event("Say the single word hello."))
        await conn.send_event({"type": "response.create"})

        events = await conn.collect_until(is_type("response.done"), timeout=60)
        types = {e.type for e in events}
        assert "response.created" in types
        assert "response.output_item.added" in types
        assert "response.done" in types

        transcript = response_transcript(events)
        assert transcript.strip() != "", "no assistant text reconstructed from deltas"

        done = last_of_type(events, "response.done")
        assert done is not None
        assert (
            done.raw["response"].get("usage") is not None
        ), "response.done is missing normalized usage"


@pytest.mark.parametrize("provider", PROVIDERS, ids=PROVIDER_IDS)
async def test_delta_matches_final_transcript(
    provider: RealtimeProvider, proxy_ws_url: str, proxy_api_key: str
) -> None:
    """Concatenated deltas must equal the provider's final transcript field when
    one is emitted -> guards the delta-merging logic in the transform layer."""
    skip_if_creds_missing(provider)

    async with connect_realtime(
        proxy_ws_url=proxy_ws_url, api_key=proxy_api_key, model=provider.model
    ) as conn:
        await conn.collect_until(is_type("session.created"), timeout=20)
        await conn.send_event(
            {
                "type": "session.update",
                "session": {"modalities": ["text"], "instructions": "Be terse."},
            }
        )
        await conn.collect_until(is_type("session.updated"), timeout=20)

        await conn.send_event(user_text_event("Count to three."))
        await conn.send_event({"type": "response.create"})
        events = await conn.collect_until(is_type("response.done"), timeout=60)

        deltas = response_transcript(events)
        done_events = of_type(events, "response.audio_transcript.done") or of_type(
            events, "response.text.done"
        )
        if not done_events:
            pytest.skip(f"{provider.id}: no terminal transcript event to compare")
        final = done_events[-1].raw.get("transcript") or done_events[-1].raw.get("text")
        assert final is not None
        assert deltas == final, "concatenated deltas do not equal the final transcript"
