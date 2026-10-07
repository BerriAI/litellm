import json
from pathlib import Path
from typing import Final

import pytest
from pydantic import BaseModel, Field

from litellm.litellm_core_utils.payload_shapes import PayloadShape, ShapeLimits, StreamShapes, extract_shape

pytestmark: Final = pytest.mark.requires_rust_extension

_FIXTURE: Final = Path(__file__).resolve().parents[2] / "litellm-rust/crates/tracing/tests/fixtures/payload-shapes.json"


@pytest.mark.parametrize("case", json.loads(_FIXTURE.read_text()))
def test_python_and_rust_share_the_shape_contract(case):
    shape: Final = extract_shape(case["input"])
    assert list(shape.field_paths) == case["expected"]["field_paths"]
    assert shape.truncated == case["expected"]["truncated"]


@pytest.mark.parametrize("limits", (ShapeLimits(nodes=1), ShapeLimits(depth=0), ShapeLimits(paths=1), ShapeLimits(bytes=1)))
def test_limits_discard_partial_output(limits):
    assert extract_shape({"messages": [{"content": "private"}]}, limits) == PayloadShape(truncated=True)


def test_scalar_values_are_never_formatted_and_cycles_are_bounded():
    class Private:
        def __str__(self):
            pytest.fail("scalar must not be formatted")

        def __repr__(self):
            pytest.fail("scalar must not be formatted")

    assert extract_shape({"content": Private()}) == PayloadShape(("$['content']",))
    cyclic: Final = []
    cyclic.append(cyclic)
    assert extract_shape(cyclic).truncated


def test_model_aliases_exclusions_and_extra_fields_without_dumping_values():
    class Response(BaseModel):
        model_config = {"extra": "allow"}
        text: str = Field(serialization_alias="content")
        hidden: str = Field(exclude=True)

    response: Final = Response(text="private", hidden="secret", usage={"tokens": 1})
    assert extract_shape(response).field_paths == ("$['content']", "$['usage']", "$['usage']['tokens']")


class Capture:
    def __init__(self):
        self.events: tuple[object, ...] = ()

    def emit_payload_shape(self, event: str) -> None:
        self.events += (json.loads(event),)


@pytest.mark.parametrize("outcome", ("success", "failure", "cancelled"))
def test_stream_summaries_merge_shapes_once_and_never_copy_values(outcome):
    capture: Final = Capture()
    stream: Final = StreamShapes("call-1", native=capture)
    stream.add("provider.response.received", 'data: {"delta":{"text":"private"}}\n\n')
    stream.add("provider.response.received", {"usage": {"tokens": 2}})
    stream.add("litellm.response.normalized", {"choices": [{"delta": {"content": "private"}}]})
    stream.finish(outcome)
    stream.finish("cancelled")
    assert len(capture.events) == 2
    received, normalized = capture.events
    assert received["shape"]["field_paths"] == ["$['delta']", "$['delta']['text']", "$['usage']", "$['usage']['tokens']"]
    assert normalized["shape"]["field_paths"] == ["$['choices']", "$['choices'][*]['delta']", "$['choices'][*]['delta']['content']"]
    assert received["capture_id"] == normalized["capture_id"] == "call-1"
    assert received["outcome"] == normalized["outcome"] == outcome
    assert "private" not in json.dumps(capture.events)


def test_unparseable_stream_data_marks_summary_incomplete():
    capture: Final = Capture()
    stream: Final = StreamShapes("call-1", native=capture)
    stream.add("provider.response.received", b'data: {"partial":')
    stream.finish("cancelled")
    assert capture.events[0]["shape"] == {"field_paths": [], "truncated": True}


def test_native_extraction_does_not_serialize_models_or_copy_scalar_payloads():
    class Response(BaseModel):
        content: str

        def model_dump(self, **kwargs: object):  # kwargs-ok: accepts the Pydantic serializer contract to detect any attempted call
            pytest.fail("shape extraction must not serialize model values")

    payload: Final = Response(content="x" * 1_048_576)
    assert extract_shape({"response": payload}).field_paths == ("$['response']", "$['response']['content']")
    assert len(payload.content) == 1_048_576
