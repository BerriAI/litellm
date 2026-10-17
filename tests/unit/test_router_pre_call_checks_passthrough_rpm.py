"""
Regression tests for #45987.

router_settings.enable_pre_call_checks filters deployments that reached their
per-minute rpm budget on /v1/chat/completions (_acompletion tracks selection
usage via _track_deployment_metrics), but passthrough routes such as
/v1/messages go through _ageneric_api_call_with_fallbacks_helper, which never
recorded selection usage, so a deployment with `rpm: 1` was never filtered out
within the same UTC minute.
"""

from unittest.mock import AsyncMock, patch

import pytest

from litellm import Router


def _make_deployment(dep_id: str, rpm: int | None = None) -> dict:
    params: dict = {"model": "gpt-4o", "api_key": "key"}
    if rpm is not None:
        params["rpm"] = rpm
    return {
        "model_name": "m-group",
        "litellm_params": params,
        "model_info": {"id": dep_id},
    }


@pytest.mark.asyncio
async def test_passthrough_helper_tracks_selection_usage():
    """
    _ageneric_api_call_with_fallbacks_helper must record the picked deployment
    against its per-minute rpm budget at selection time (mirroring _acompletion).
    Before the fix the helper never called _track_deployment_metrics, so the
    deployment's local request count stayed at 0 and _pre_call_checks never
    enforced `rpm` on /v1/messages.
    """
    router = Router(model_list=[_make_deployment("dep-a", rpm=1)])

    async def fake_original(**kwargs):
        return {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}

    with patch.object(
        router,
        "async_get_available_deployment",
        AsyncMock(return_value=_make_deployment("dep-a", rpm=1)),
    ):
        await router._ageneric_api_call_with_fallbacks_helper(
            model="m-group",
            original_generic_function=fake_original,
            messages=[{"role": "user", "content": "hi"}],
        )

    count = router.cache.get_cache(key="dep-a", local_only=True)
    assert count == 1


@pytest.mark.asyncio
async def test_pre_call_checks_filters_deployment_at_rpm_limit_after_tracking():
    """
    End-to-end rpm accounting: once selection usage has been recorded against a
    deployment (as the fixed helper now does), a follow-up _pre_call_checks pass
    must drop that deployment when its count reached its rpm budget, so repeated
    /v1/messages requests stop hitting an `rpm: 1` deployment within the same
    UTC minute.
    """
    router = Router(
        model_list=[
            _make_deployment("dep-a", rpm=1),
            _make_deployment("dep-b"),
        ]
    )

    healthy = [_make_deployment("dep-a", rpm=1), _make_deployment("dep-b")]

    # First pass: dep-a has not been used yet -> kept.
    kept = router._pre_call_checks(
        model="m-group",
        healthy_deployments=healthy,
        messages=[{"role": "user", "content": "hi"}],
    )
    assert any(d["model_info"]["id"] == "dep-a" for d in kept)

    # Record one request against dep-a (what the fixed helper does on selection).
    router._track_deployment_metrics(deployment=_make_deployment("dep-a", rpm=1), parent_otel_span=None)

    # Second pass: dep-a is at its rpm budget -> filtered out, dep-b remains.
    kept2 = router._pre_call_checks(
        model="m-group",
        healthy_deployments=healthy,
        messages=[{"role": "user", "content": "hi"}],
    )
    ids2 = [d["model_info"]["id"] for d in kept2]
    assert "dep-a" not in ids2
    assert "dep-b" in ids2
