"""
Direct unit tests for the signature codec used to round-trip OpenAI Responses
reasoning items through Anthropic's `thinking.signature` field.

Covers both _pack_signature and _unpack_signature including the malformed-input
except branch, the non-prefixed signature degrade path (mid-conversation model
switch from chatgpt/openai to native anthropic), and the empty-input
short-circuit paths.
"""

import base64

from litellm.llms.anthropic.experimental_pass_through.responses_adapters._signature_codec import (
    _SIG_PREFIX,
    _pack_signature,
    _unpack_signature,
)


def test_pack_signature_none_encrypted_returns_none():
    assert _pack_signature("rs_x", None) is None


def test_pack_signature_empty_encrypted_returns_none():
    assert _pack_signature("rs_x", "") is None


def test_pack_signature_round_trip():
    rs_id = "rs_abc123"
    enc = "ENCRYPTED-BLOB-XYZ=="
    packed = _pack_signature(rs_id, enc)

    assert packed is not None
    assert packed.startswith(_SIG_PREFIX)
    assert _unpack_signature(packed) == (rs_id, enc)


def test_pack_signature_none_id_still_packs():
    """OpenAI sometimes emits reasoning items without an id; encrypted_content
    alone is enough to replay, so we still pack."""
    packed = _pack_signature(None, "ENC")
    assert packed is not None
    assert _unpack_signature(packed) == (None, "ENC")


def test_unpack_signature_none_returns_pair_of_nones():
    assert _unpack_signature(None) == (None, None)


def test_unpack_signature_empty_returns_pair_of_nones():
    assert _unpack_signature("") == (None, None)


def test_unpack_signature_anthropic_native_signature_degrades():
    """A native Anthropic signature lacks our prefix — degrade silently so a
    mid-conversation model switch back to anthropic doesn't crash. We return
    (None, None) and skip the reasoning replay; the conversation still works,
    just without reasoning continuity."""
    native = "EpYBChIQAAAAAA=="  # plausible-looking native anthropic signature
    assert _unpack_signature(native) == (None, None)


def test_unpack_signature_malformed_base64_handled():
    """Covers the `except Exception` fallback in _unpack_signature."""
    bad = _SIG_PREFIX + "not-base64!!!@@@"
    assert _unpack_signature(bad) == (None, None)


def test_unpack_signature_valid_base64_but_invalid_json_handled():
    """Covers the json.loads failure inside the except branch."""
    garbage = _SIG_PREFIX + base64.b64encode(b"not json at all").decode()
    assert _unpack_signature(garbage) == (None, None)


def test_unpack_signature_valid_base64_json_without_expected_keys():
    """Codec ignores extra keys and tolerates missing keys via .get()."""
    payload = base64.b64encode(b'{"unrelated":"value"}').decode()
    rs_id, enc = _unpack_signature(_SIG_PREFIX + payload)
    assert rs_id is None
    assert enc is None
