"""
Purpose
- Lock minimal helpers in extras.response_utils (content extraction, stream assembly, cost augmentation).

Scope
- DOES: assert dict/object handling and simple stream concatenation.
- DOES NOT: contact providers or depend on models.

Run
- `pytest tests/local_testing -k test_response_utils -q`
"""
import asyncio
import types


def test_extract_content_dict_and_object():
    from litellm.extras.response_utils import extract_content

    d = {"choices": [{"message": {"role": "assistant", "content": "HI"}}]}
    assert extract_content(d) == "HI"

    class Msg: role="assistant"; content="OK"
    class Choice: pass
    c = Choice(); c.message = Msg()
    class Resp: pass
    r = Resp(); r.choices = [c]
    assert extract_content(r) == "OK"


def test_assemble_stream_text():
    from litellm.extras.response_utils import assemble_stream_text

    async def agen():
        yield {"choices": [{"delta": {"content": "A"}}]}
        yield {"choices": [{"delta": {"content": "B"}}]}
    out = asyncio.run(assemble_stream_text(agen()))
    assert out == "AB"


def test_augment_json_with_cost():
    from litellm.extras.response_utils import augment_json_with_cost

    class Resp:
        usage = {"prompt_tokens": 1, "completion_tokens": 2}
        _hidden_params = {"cache_hit": True}

    j = '{"answer":"ok"}'
    out = augment_json_with_cost(j, Resp())
    assert '"metadata"' in out
