"""The v1/v2/fast-path fork is a pure decision, so it is tested as one."""

from litellm.translation import Route, route

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
