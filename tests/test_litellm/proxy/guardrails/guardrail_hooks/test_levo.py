from types import SimpleNamespace

import pytest

from litellm.proxy.guardrails.guardrail_hooks.generic_guardrail_api.generic_guardrail_api import (
    GenericGuardrailAPI,
)
from litellm.proxy.guardrails.guardrail_hooks.levo import initialize_guardrail
from litellm.proxy.guardrails.guardrail_hooks.levo.levo import (
    LEVO_GUARDRAIL_PATH,
    LevoGuardrail,
)
from litellm.proxy.guardrails.guardrail_registry import (
    guardrail_class_registry,
    guardrail_initializer_registry,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.proxy.guardrails.guardrail_hooks.levo import (
    LevoGuardrailConfigModel,
)


def _params(**overrides: object) -> SimpleNamespace:
    """LitellmParams-shaped stub, as the proxy passes to the initializer."""
    base = dict(
        guardrail="levo",
        mode=["pre_call", "post_call"],
        api_base="http://levo-gateway:8080",
        api_key="s3cret",
        default_on=True,
        optional_params=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _guardrail(name: str = "levo") -> dict[str, str]:
    return {"guardrail_name": name}


def test_registry_membership():
    assert "levo" in guardrail_initializer_registry
    assert guardrail_class_registry["levo"] is LevoGuardrail


def test_config_model_wiring():
    assert LevoGuardrail.get_config_model() is LevoGuardrailConfigModel
    assert LevoGuardrailConfigModel.ui_friendly_name() == "Levo AI Gateway"


def test_supported_hooks_limited_to_pre_and_post():
    # during_call would duplicate the pre_call input event without adding a
    # decision point.
    assert LevoGuardrail.get_supported_event_hooks() == [
        GuardrailEventHooks.pre_call,
        GuardrailEventHooks.post_call,
    ]


def test_endpoint_path_appended_to_api_base():
    g = LevoGuardrail(api_base="http://levo-gateway:8080", guardrail_name="levo")
    assert g.api_base == f"http://levo-gateway:8080{LEVO_GUARDRAIL_PATH}"


def test_api_key_sent_as_x_api_key():
    # The gateway serves this endpoint on its data-plane port and rejects
    # unauthenticated callers, so the shared secret must reach it.
    g = LevoGuardrail(api_base="http://levo-gateway:8080", api_key="s3cret", guardrail_name="levo")
    assert g.headers.get("x-api-key") == "s3cret"


def test_initializer_requires_api_base():
    with pytest.raises(ValueError, match="api_base is required"):
        initialize_guardrail(_params(api_base=None), _guardrail())


def test_initializer_builds_working_callback():
    cb = initialize_guardrail(_params(), _guardrail())
    assert isinstance(cb, LevoGuardrail)
    assert cb.default_on is True
    assert cb.api_base.endswith(LEVO_GUARDRAIL_PATH)


def test_initializer_reads_optional_params_flattened_like_ui():
    # The UI submits provider settings under optional_params rather than at the
    # top level; both shapes must reach the constructor.
    cb = initialize_guardrail(
        _params(optional_params={"unreachable_fallback": "fail_open"}),
        _guardrail(),
    )
    assert cb.unreachable_fallback == "fail_open"


def test_unreachable_fallback_defaults_to_fail_closed():
    cb = initialize_guardrail(_params(), _guardrail())
    assert cb.unreachable_fallback == "fail_closed"


# ── streaming ───────────────────────────────────────────────────────────────
#
# The reason this integration exists separately from generic_guardrail_api.


def test_streaming_buffered_by_default():
    # A response-side block is only meaningful if it lands before the client
    # sees the content. Without buffering, chunks are emitted as they are
    # produced and a violation is detected after the fact.
    g = LevoGuardrail(api_base="http://levo-gateway:8080", guardrail_name="levo")
    assert g.streaming_buffer_until_moderated is True
    assert g.streaming_end_of_stream_only is True


def test_streaming_buffering_can_be_disabled():
    # Operators who need time-to-first-token more than response-side
    # enforcement can opt out.
    g = LevoGuardrail(
        api_base="http://levo-gateway:8080",
        guardrail_name="levo",
        buffer_streaming_until_moderated=False,
    )
    assert g.streaming_buffer_until_moderated is False


def test_streaming_flag_settable_via_optional_params():
    cb = initialize_guardrail(
        _params(optional_params={"buffer_streaming_until_moderated": False}),
        _guardrail(),
    )
    assert cb.streaming_buffer_until_moderated is False


def test_apply_guardrail_defined_on_the_class_not_inherited():
    """Regression: the proxy selects the unified guardrail path with
    ``"apply_guardrail" in type(callback).__dict__``, which inspects the
    class's own attributes and does not see inherited methods.

    A subclass that relies on inheritance is constructed, registered and even
    consulted via ``should_run_guardrail`` — but never invoked, so every
    request passes unscanned while the guardrail reports healthy. Guard the
    binding so that failure mode cannot return silently.
    """
    assert "apply_guardrail" in LevoGuardrail.__dict__


def test_apply_guardrail_is_the_base_method_not_a_second_wrapper():
    """Regression: the binding above must alias the base method rather than
    wrap it in another ``@log_guardrail_information`` layer.

    Two decorated layers log the call twice — the inner wrapper's ``finally``
    resets the "already recorded" ContextVar to the value the outer wrapper
    set, so the outer sees an unrecorded call and emits its own span, Datadog
    record and spend-log entry on top of the inner one.
    """
    assert LevoGuardrail.apply_guardrail is GenericGuardrailAPI.apply_guardrail
