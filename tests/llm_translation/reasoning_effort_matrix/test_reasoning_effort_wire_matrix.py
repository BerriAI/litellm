"""reasoning_effort wire-translation matrix.

Automated successor to the manual 231-cell QA sweep behind #27039 / #27074.

Two always-on, fully-offline oracles (the per-PR regression net):

* ``test_wire_request`` — for every (route, entrypoint, model, effort) cell,
  assert the reasoning subtree LiteLLM builds on the wire matches the
  hand-authored spec *and* its presence/absence (catches silent strips), or
  that a client-rejected effort raises a clean 400 (catches the
  #27074 ValueError->BadRequest fix).

* ``test_no_unmapped_reasoning_models`` — staleness canary; fails loudly when
  model_prices grows a reasoning-capable Claude family the matrix doesn't
  cover (the #27039 -> #27074 per-model recurrence).

One CI-gated, VCR-recorded oracle (dormant off-CI):

* ``test_provider_accepts_built_request`` — replays/records the real Anthropic
  round-trip for wire cells and asserts Anthropic accepted the request
  LiteLLM built. Skipped locally; the offline oracles are the local signal.
"""

from __future__ import annotations

import pytest

import litellm

from .conftest import ci_record_enabled
from .spec import (
    CHAT,
    CLIENT_400,
    MESSAGES,
    OMIT,
    Cell,
    Expected,
    all_cells,
    build_wire_request,
    is_clean_400,
    unmapped_reasoning_claude_families,
)

_CELLS = all_cells()
_IDS = [c.id for c in _CELLS]


@pytest.mark.parametrize("cell", _CELLS, ids=_IDS)
def test_wire_request(cell: Cell):
    """Offline: the wire subtree (or clean-400) matches the spec exactly."""
    expected = cell.expected

    if expected == CLIENT_400:
        with pytest.raises(BaseException) as ei:  # noqa: PT011 - we classify below
            build_wire_request(cell.route, cell.entrypoint, cell.model, cell.effort)
        assert is_clean_400(ei.value), (
            f"{cell.id}: expected a clean 400-class rejection before the wire, "
            f"got {type(ei.value).__name__}: {ei.value!r}. "
            f"A bare ValueError/500 here is the exact #27074 regression."
        )
        return

    assert isinstance(expected, Expected)
    built = build_wire_request(cell.route, cell.entrypoint, cell.model, cell.effort)

    # Value AND presence/absence (None == key must be absent).
    assert built["thinking"] == expected.thinking, (
        f"{cell.id}: thinking mismatch — expected {expected.thinking!r}, "
        f"built {built['thinking']!r}"
    )
    assert built["output_config"] == expected.output_config, (
        f"{cell.id}: output_config mismatch — expected "
        f"{expected.output_config!r}, built {built['output_config']!r}"
    )


def test_no_unmapped_reasoning_models():
    """Staleness canary — see spec.unmapped_reasoning_claude_families."""
    unmapped = unmapped_reasoning_claude_families()
    assert not unmapped, (
        "model_prices has reasoning-capable Claude families not covered by the "
        f"matrix: {unmapped}. Add each to spec.MODEL_TIER with its tier (and "
        "expected column), or to CANARY_ALLOWLIST if intentionally excluded."
    )


# --------------------------------------------------------------------------- #
# CI-gated live provider-acceptance (VCR-recorded; dormant off-CI)
# --------------------------------------------------------------------------- #

_LIVE_TRANSIENT = (
    litellm.ServiceUnavailableError,
    litellm.Timeout,
    litellm.RateLimitError,
    litellm.InternalServerError,
)

# Scope the live half to the Anthropic-direct route: ANTHROPIC_API_KEY is the
# one provider credential reliably present in the llm_translation CI job.
# Bedrock/Vertex/Azure live coverage is a documented future extension.
_LIVE_CELLS = [
    c
    for c in _CELLS
    if c.route == "anthropic"
    and not isinstance(c.expected, str)  # wire cells only (client-400 done offline)
]
_LIVE_IDS = [c.id for c in _LIVE_CELLS]


@pytest.mark.skipif(
    not ci_record_enabled(),
    reason="live provider-acceptance is CI/VCR-only (set CASSETTE_REDIS_URL "
    "or MATRIX_FORCE_LIVE=1)",
)
@pytest.mark.parametrize("cell", _LIVE_CELLS, ids=_LIVE_IDS)
def test_provider_accepts_built_request(cell: Cell):
    """CI: Anthropic must accept the request LiteLLM builds for this cell.

    The recorded response is the proof oracle (b): a non-2xx here means
    LiteLLM built a payload Anthropic rejects — a real regression, not flake.
    Transient infra errors are skipped (repo convention).
    """
    kwargs = {
        "model": f"anthropic/{cell.model}",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "max_tokens": 16,
    }
    if cell.effort != OMIT:
        kwargs["reasoning_effort"] = cell.effort

    try:
        if cell.entrypoint == CHAT:
            resp = litellm.completion(**kwargs)
        else:  # MESSAGES
            resp = litellm.anthropic_messages(
                model=kwargs["model"],
                messages=kwargs["messages"],
                max_tokens=kwargs["max_tokens"],
                **({"reasoning_effort": cell.effort} if cell.effort != OMIT else {}),
            )
    except _LIVE_TRANSIENT as e:
        pytest.skip(f"transient provider error: {type(e).__name__}: {e}")
    except litellm.BadRequestError as e:
        pytest.fail(
            f"{cell.id}: Anthropic rejected the request LiteLLM built "
            f"(400). LiteLLM mistranslated this cell. {e}"
        )

    assert resp is not None, f"{cell.id}: no response from Anthropic"
