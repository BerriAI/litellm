import litellm
import pytest

@pytest.mark.asyncio
async def test_compaction_noop_under_threshold():

    """Case 1: small conversation, high threshold → no compaction item"""

    response = await litellm.aresponses(

        model="claude-haiku-4-5-20251001",

        input=[

            {"type": "message", "role": "user", "content": "Help me debug a production incident."},

            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "What symptoms are you seeing?"}]},

            {"type": "message", "role": "user", "content": "We are seeing intermittent 502s from one provider path."},

        ],

        context_management=[{"type": "compaction", "compact_threshold": 200000}],

        store=False,

        max_output_tokens=200,

    )

    assert response is not None

    output_types = [item.get("type") for item in response.output]

    assert "compaction" not in output_types

@pytest.mark.asyncio
async def test_compaction_triggers_over_threshold():

    """Case 2: large conversation, low threshold → compaction item is last in output"""

    fat_message = "word " * 5000

    response = await litellm.aresponses(

        model="claude-haiku-4-5-20251001",

        input=[

            {"type": "message", "role": "user", "content": "Help me debug a production incident."},

            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": fat_message}]},

            {"type": "message", "role": "user", "content": "We are seeing intermittent 502s from one provider path."},

        ],

        context_management=[{"type": "compaction", "compact_threshold": 100}],

        store=False,

        max_output_tokens=200,

    )

    assert response is not None

    assert response.output[-1].get("type") == "compaction"

    assert response.output[-1].get("encrypted_content")


@pytest.mark.asyncio

async def test_no_recompaction_when_under_threshold():

    """Case 3: prior compaction item + short message, under threshold → no new compaction"""

    import base64

    prior_cmp = {

        "type": "compaction",

        "id": "cmp_prior_001",

        "encrypted_content": base64.b64encode(b"Previous conversation summary.").decode(),

    }

    response = await litellm.aresponses(

        model="claude-haiku-4-5-20251001",

        input=[

            prior_cmp,

            {"type": "message", "role": "user", "content": "What was the main issue?"},

        ],

        context_management=[{"type": "compaction", "compact_threshold": 200000}],

        store=False,

        max_output_tokens=100,

    )

    assert response is not None

    new_cmp = [i for i in response.output if i.get("type") == "compaction" and i.get("id") != "cmp_prior_001"]

    assert not new_cmp