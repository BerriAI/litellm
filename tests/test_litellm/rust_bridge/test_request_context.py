from types import SimpleNamespace

from litellm.rust_bridge.request import NativeRequestCapabilities, request_context


def test_request_context_preserves_identity_attribution_and_capabilities() -> None:
    capabilities = NativeRequestCapabilities(execution_mode="async", stream=True)

    context = request_context(
        logging_obj=SimpleNamespace(litellm_call_id="call-1", litellm_trace_id="trace-1"),
        request_model="router-alias",
        litellm_params={
            "metadata": {
                "user_api_key_hash": "hash-1",
                "user_api_key_user_id": "user-1",
                "user_api_key_team_id": "team-1",
            }
        },
        capabilities=capabilities,
    )

    assert context.litellm_call_id == "call-1"
    assert context.trace_id == "trace-1"
    assert context.request_model == "router-alias"
    assert context.attribution.user_api_key_hash == "hash-1"
    assert context.attribution.user_api_key_user_id == "user-1"
    assert context.attribution.user_api_key_team_id == "team-1"
    assert context.capabilities is capabilities


def test_request_context_ignores_untyped_identity_values() -> None:
    context = request_context(
        logging_obj=SimpleNamespace(litellm_call_id=1, litellm_trace_id=[]),
        request_model="model",
        litellm_params={"user_api_key_user_id": 42},
    )

    assert context.litellm_call_id is None
    assert context.trace_id is None
    assert context.attribution.user_api_key_user_id is None
