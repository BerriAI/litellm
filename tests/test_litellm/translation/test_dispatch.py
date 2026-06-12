"""The v1/v2/fast-path fork is a pure decision, so it is tested as one.

The never-port encoding (researcher-4's proposal) lives here too: the 23
permanent-fallback names are CODE (``NeverPortProvider`` + ``NEVER_PORT``),
the disjointness test makes promotion a deliberate two-file edit, and the
seam-level test proves each name falls back typed, never raising.
"""

from typing import cast, get_args

from litellm.translation import Route, route
from litellm.translation.dispatch import NEVER_PORT, Provider
from litellm.translation.engine import pipeline

_ALL = frozenset({"anthropic", "openai_compat"})


def test_unopted_provider_stays_on_v1() -> None:
    decision = route(
        schema="anthropic_messages",
        provider="anthropic",
        enabled_providers=frozenset(),
        body_touching=False,
    )
    assert decision.tag == "v1"


def test_same_family_takes_the_fast_path() -> None:
    decision = route(
        schema="anthropic_messages",
        provider="anthropic",
        enabled_providers=_ALL,
        body_touching=False,
    )
    assert decision.tag == "fast_path"
    assert decision.fast_path == "anthropic"


def test_openai_into_openai_compat_is_same_family() -> None:
    decision = route(
        schema="openai_chat",
        provider="openai_compat",
        enabled_providers=_ALL,
        body_touching=False,
    )
    assert decision == Route.of_fast_path("openai_compat")


def test_body_touching_feature_drops_off_the_fast_path() -> None:
    decision = route(
        schema="anthropic_messages",
        provider="anthropic",
        enabled_providers=_ALL,
        body_touching=True,
    )
    assert decision.tag == "v2"
    assert decision.v2 == "anthropic"


def test_cross_family_goes_through_the_ir() -> None:
    decision = route(
        schema="openai_chat",
        provider="anthropic",
        enabled_providers=_ALL,
        body_touching=False,
    )
    assert decision == Route.of_v2("anthropic")


# ---------------------------------------------------------------------------
# Never-port providers as code (researcher-4's encoding)
# ---------------------------------------------------------------------------


def test_never_port_is_disjoint_from_registered_providers() -> None:
    """Promoting a never-port provider must be a deliberate two-file edit
    (remove from NeverPortProvider, add to Provider) — adding it to Provider
    alone goes red here."""
    assert len(NEVER_PORT) == 23
    assert NEVER_PORT.isdisjoint(set(get_args(Provider)))
    # deliberately OUTSIDE the policy list (each carries its own mechanism):
    # baseten keeps its re-evaluate canary; cohere's never-port surface is a
    # ROUTE ("v1/" in the model name), not a provider — that predicate
    # belongs inside the future cohere module (researcher-4 Part 2); the
    # wave-1b drops (aiml + the 7 non-enum JSON providers) keep their own
    # re-evaluate canaries.
    assert "baseten" not in NEVER_PORT
    assert "cohere" not in NEVER_PORT
    assert "cohere_chat" not in NEVER_PORT
    assert "aiml" not in NEVER_PORT
    assert "veniceai" not in NEVER_PORT


def test_never_port_providers_have_no_v2_registration() -> None:
    for provider in sorted(NEVER_PORT):
        assert provider not in pipeline._SERIALIZERS, provider
        assert provider not in pipeline._RESPONSE_PARSERS, provider
        assert provider not in pipeline._RESPONSE_DIALECTS, provider
        assert provider not in pipeline._RAW_GUARDS, provider


def test_never_port_providers_fall_back_typed_at_the_seam() -> None:
    """The seam-level proof: translate for a never-port provider yields the
    typed "no v2 chat serializer" fallback (a value, never an exception),
    and route() keeps it on v1 even with every registered provider
    enabled."""
    from litellm.translation import translate_chat_request
    from litellm.translation_seam import build_translation_deps

    deps = build_translation_deps()
    raw = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    every_registered = frozenset(get_args(Provider))
    for provider in sorted(NEVER_PORT):
        # deliberately illegal input: the negative path NEEDS a never-port
        # name where a Provider is expected; cast carries that intent
        # without masking real arg-type regressions (critic-wave1b N4)
        illegal = cast(Provider, provider)
        result = translate_chat_request(raw, illegal, deps)
        assert result.is_error(), provider
        assert "no v2 chat serializer" in result.error.summary, provider
        decision = route(
            schema="openai_chat",
            provider=illegal,
            enabled_providers=every_registered,
            body_touching=False,
        )
        assert decision.tag == "v1", provider
