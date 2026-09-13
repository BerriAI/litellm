"""
Regression tests for the clientside-credential fallback re-validation scope
lookup (see litellm/router_utils/clientside_credential_handler.py).

The proxy stamps the opt-in scope into exactly one metadata bucket
(``_get_metadata_variable_name``): ``litellm_metadata`` on
LITELLM_METADATA_ROUTES (/v1/messages, responses, batches, bedrock, files)
and the thread/assistant routes, ``metadata`` everywhere else — while a
caller-supplied provider-facing ``metadata`` object can survive as a second
kwargs bucket on those routes. The strip helper must consult EVERY bucket so
an unstamped caller bucket cannot shadow the proxy stamp (PR 40001 review:
"High: Duplicate metadata bypasses fallback credential stripping").
"""

from litellm.router_utils.clientside_credential_handler import (
    PROXY_CLIENTSIDE_CREDENTIAL_SCOPE_METADATA_KEY,
    strip_clientside_credentials_without_deployment_opt_in,
)

SCOPE_KEY = PROXY_CLIENTSIDE_CREDENTIAL_SCOPE_METADATA_KEY
ATTACKER_API_BASE = "http://127.0.0.1:9/attacker"

DEPLOYMENT_WITHOUT_OPT_IN = {
    "model_name": "model-b",
    "litellm_params": {
        "model": "openai/gpt-4o",
        "api_key": "sk-backend-b",
        "api_base": "http://legit-upstream",
    },
}
DEPLOYMENT_WITH_OPT_IN = {
    "model_name": "model-b",
    "litellm_params": {
        "model": "openai/gpt-4o",
        "api_key": "sk-backend-b",
        "api_base": "http://legit-upstream",
        "configurable_clientside_auth_params": ["api_base"],
    },
}


def _kwargs_with_caller_credential() -> dict:
    return {"api_base": ATTACKER_API_BASE, "messages": [{"role": "user", "content": "hi"}]}


def test_caller_metadata_does_not_shadow_per_model_stamp_in_litellm_metadata():
    """The reported bypass shape: unstamped caller ``metadata`` object + per_model
    stamp in ``litellm_metadata`` (LITELLM_METADATA_ROUTES / thread routes)."""
    kwargs = _kwargs_with_caller_credential()
    kwargs["metadata"] = {"user_id": "legit-anthropic-metadata"}
    kwargs["litellm_metadata"] = {SCOPE_KEY: "per_model"}

    strip_clientside_credentials_without_deployment_opt_in(deployment=DEPLOYMENT_WITHOUT_OPT_IN, kwargs=kwargs)

    assert "api_base" not in kwargs


def test_reverse_bucket_order_still_strips():
    """Stamp in ``metadata`` (regular routes) must not be shadowed by an
    unstamped ``litellm_metadata`` bucket either."""
    kwargs = _kwargs_with_caller_credential()
    kwargs["metadata"] = {SCOPE_KEY: "per_model"}
    kwargs["litellm_metadata"] = {"user_id": "unrelated"}

    strip_clientside_credentials_without_deployment_opt_in(deployment=DEPLOYMENT_WITHOUT_OPT_IN, kwargs=kwargs)

    assert "api_base" not in kwargs


def test_json_string_metadata_bucket_does_not_shadow_stamp():
    """A JSON-encoded string bucket (multipart/extra_body shape) cannot carry
    the stamp and must not shadow the dict bucket that does."""
    import json

    kwargs = _kwargs_with_caller_credential()
    kwargs["metadata"] = json.dumps({"user_id": "legit"})
    kwargs["litellm_metadata"] = {SCOPE_KEY: "per_model"}

    strip_clientside_credentials_without_deployment_opt_in(deployment=DEPLOYMENT_WITHOUT_OPT_IN, kwargs=kwargs)

    assert "api_base" not in kwargs


def test_conflicting_scopes_fail_closed_to_per_model():
    """If buckets ever disagree, the most restrictive scope wins. (Caller-forged
    scope values are stripped upstream, so this is defense in depth.)"""
    kwargs = _kwargs_with_caller_credential()
    kwargs["metadata"] = {SCOPE_KEY: "proxy_wide"}
    kwargs["litellm_metadata"] = {SCOPE_KEY: "per_model"}

    strip_clientside_credentials_without_deployment_opt_in(deployment=DEPLOYMENT_WITHOUT_OPT_IN, kwargs=kwargs)

    assert "api_base" not in kwargs


def test_bracket_encoded_and_case_variant_keys_do_not_shadow_stamp():
    """Form-style ``metadata[...]`` keys and case variants are distinct kwargs
    keys and must never be mistaken for (or shadow) a stamped bucket."""
    kwargs = _kwargs_with_caller_credential()
    kwargs["metadata[foo]"] = "bar"
    kwargs["Metadata"] = {"user_id": "case-variant"}
    kwargs["litellm_metadata"] = {SCOPE_KEY: "per_model"}

    strip_clientside_credentials_without_deployment_opt_in(deployment=DEPLOYMENT_WITHOUT_OPT_IN, kwargs=kwargs)

    assert "api_base" not in kwargs


def test_no_scope_stamp_leaves_sdk_router_behavior_unchanged():
    """Plain SDK completion call (no proxy stamp anywhere): the per-call
    clientside credential feature must keep working."""
    kwargs = _kwargs_with_caller_credential()
    kwargs["metadata"] = {"user_id": "sdk-user"}

    strip_clientside_credentials_without_deployment_opt_in(deployment=DEPLOYMENT_WITHOUT_OPT_IN, kwargs=kwargs)

    assert kwargs["api_base"] == ATTACKER_API_BASE


def test_missing_metadata_entirely_leaves_behavior_unchanged():
    kwargs = _kwargs_with_caller_credential()

    strip_clientside_credentials_without_deployment_opt_in(deployment=DEPLOYMENT_WITHOUT_OPT_IN, kwargs=kwargs)

    assert kwargs["api_base"] == ATTACKER_API_BASE


def test_proxy_wide_scope_leaves_credential_untouched():
    """Admin opted every deployment in via
    general_settings.allow_client_side_credentials."""
    kwargs = _kwargs_with_caller_credential()
    kwargs["metadata"] = {SCOPE_KEY: "proxy_wide"}

    strip_clientside_credentials_without_deployment_opt_in(deployment=DEPLOYMENT_WITHOUT_OPT_IN, kwargs=kwargs)

    assert kwargs["api_base"] == ATTACKER_API_BASE


def test_opted_in_deployment_keeps_caller_credential_under_per_model():
    """The benign primary-path case: the dispatched deployment opted in to the
    credential key, so the caller's value is honored even under per_model."""
    kwargs = _kwargs_with_caller_credential()
    kwargs["metadata"] = {"user_id": "legit"}
    kwargs["litellm_metadata"] = {SCOPE_KEY: "per_model"}

    strip_clientside_credentials_without_deployment_opt_in(deployment=DEPLOYMENT_WITH_OPT_IN, kwargs=kwargs)

    assert kwargs["api_base"] == ATTACKER_API_BASE
