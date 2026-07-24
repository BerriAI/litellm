import pytest

from litellm.llms.vertex_ai.interactions_passthrough.id_codec import (
    VertexInteractionId,
    decode,
    encode,
    is_encoded,
)


@pytest.mark.parametrize(
    "project, location, raw_id",
    [
        ("gemini-0610-462508", "global", "video-43dffcd7-2f1f-4dc1-ac05-a8b885f8822d"),
        ("proj-2", "us-central1", "resp_bGl0ZWxsbTpzb21ldGhpbmc"),
        ("p", "global", "id;with;semicolons"),
    ],
)
def test_round_trip(project, location, raw_id):
    encoded = encode(project, location, raw_id)
    decoded = decode(encoded)
    assert decoded == VertexInteractionId(project=project, location=location, raw_id=raw_id)


def test_encoding_is_deterministic():
    a = encode("proj", "global", "video-abc")
    b = encode("proj", "global", "video-abc")
    assert a == b


def test_encoded_id_has_no_padding_and_is_urlsafe():
    encoded = encode("proj", "global", "video-abc")
    assert "=" not in encoded
    assert "/" not in encoded and "+" not in encoded


@pytest.mark.parametrize(
    "value",
    [
        "video-43dffcd7-2f1f-4dc1-ac05-a8b885f8822d",  # raw vertex id
        "resp_abc",  # raw sync id
        "",  # empty
        "not base64 at all !!!",
    ],
)
def test_decode_rejects_non_our_ids(value):
    assert decode(value) is None
    assert is_encoded(value) is False


def test_decode_rejects_openai_passthrough_managed_id():
    import base64

    # OpenAI/Azure passthrough codec uses the "passthrough" discriminator.
    plaintext = "litellm_proxy:passthrough;provider:openai;unified_id,u1;raw_id,batch_x"
    other = base64.urlsafe_b64encode(plaintext.encode()).decode().rstrip("=")
    assert decode(other) is None


def test_decode_rejects_non_string():
    assert decode(None) is None  # type: ignore[arg-type]
    assert decode(123) is None  # type: ignore[arg-type]


def test_is_encoded_true_for_our_ids():
    assert is_encoded(encode("proj", "global", "video-abc")) is True


def _b64(plaintext: str) -> str:
    import base64

    return base64.urlsafe_b64encode(plaintext.encode()).decode().rstrip("=")


def test_decode_rejects_correct_head_but_too_few_fields():
    # Correct discriminator head, but the payload has fewer than the 3 expected
    # ';'-separated fields, so the unpacking split raises ValueError -> None.
    assert decode(_b64("litellm_proxy:vertex_interaction;project,p")) is None


def test_decode_rejects_correct_head_but_wrong_field_prefixes():
    # Three fields, correct head, but the field names are not project/location/raw_id.
    assert decode(_b64("litellm_proxy:vertex_interaction;proj,p;loc,l;rid,r")) is None
