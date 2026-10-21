import asyncio
import hashlib
import json
from pathlib import Path
from typing import Final

import httpx

from poc.heldout_baseline import (
    CaptureClient,
    GatewayJevClassifierClient,
    active_tiers,
    choose_baseline_route,
    source_config,
    source_router,
)
from poc.heldout_live import _image_overlay_ids
from poc.policy import SpendSuccessCell, build_spend_success_map, choose_capability_model
from poc.run_heldout import balanced_schedule, classifier_plus_solver_cost

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("haiku-4.5", "sonnet-5", "opus-5")


def _record(task: str, model: str, category: str, cost: float, success: bool) -> dict[str, object]:
    return {
        "task_id": task,
        "model": model,
        "category_jev": category,
        "cost_usd": cost,
        "success": success,
        "error_class": "solver",
    }


def test_spend_per_success_can_choose_more_expensive_but_more_successful_model() -> None:
    records: Final = (
        _record("h", "haiku-4.5", "repository_debugging", 1.0, True),
        *(
            _record(f"s{index}", "sonnet-5", "repository_debugging", 0.8, True)
            for index in range(5)
        ),
    )
    score_map: Final = build_spend_success_map(records, MODELS)

    decision = choose_capability_model("repository_debugging", score_map, MODELS)

    assert score_map["repository_debugging"]["sonnet-5"].total_spend_usd > 1.0
    assert decision.model == "sonnet-5"


def test_category_scores_change_the_selected_model() -> None:
    score_map: Final = build_spend_success_map(
        (
            _record("a", "haiku-4.5", "repository_debugging", 1.0, True),
            _record("b", "sonnet-5", "repository_debugging", 2.0, True),
            _record("c", "haiku-4.5", "feature_implementation", 3.0, True),
            _record("d", "sonnet-5", "feature_implementation", 1.0, True),
        ),
        MODELS,
    )

    assert choose_capability_model("repository_debugging", score_map, MODELS).model == "haiku-4.5"
    assert choose_capability_model("feature_implementation", score_map, MODELS).model == "sonnet-5"


def test_zero_success_is_none_and_sparse_cell_remains_selectable() -> None:
    score_map: Final = build_spend_success_map(
        (
            _record("one", "haiku-4.5", "repository_debugging", 0.2, True),
            _record("two", "sonnet-5", "repository_debugging", 0.1, False),
        ),
        MODELS,
    )

    assert score_map["repository_debugging"]["haiku-4.5"].sparse
    assert score_map["repository_debugging"]["sonnet-5"].score_usd_per_success is None
    assert choose_capability_model("repository_debugging", score_map, MODELS).model == "haiku-4.5"


def test_unknown_and_classifier_failure_use_global_scores() -> None:
    score_map: Final = build_spend_success_map(
        (
            _record("a", "haiku-4.5", "repository_debugging", 2.0, True),
            _record("b", "sonnet-5", "feature_implementation", 1.0, True),
        ),
        MODELS,
    )

    unknown = choose_capability_model("unknown", score_map, MODELS)
    failure = choose_capability_model("repository_debugging", score_map, MODELS, classifier_failed=True)

    assert (unknown.model, unknown.category, unknown.reason) == (
        "sonnet-5",
        "global",
        "global_fallback:unknown_category",
    )
    assert (failure.model, failure.category, failure.reason) == (
        "sonnet-5",
        "global",
        "global_fallback:classifier_failure",
    )


def test_all_zero_global_scores_are_unproven() -> None:
    score_map: Final = build_spend_success_map(
        (_record("a", model, "feature_implementation", 0.5, False) for model in MODELS),
        MODELS,
    )

    decision = choose_capability_model("feature_implementation", score_map, MODELS)

    assert decision.model is None
    assert decision.reason == "unproven"


def test_heldout_manifest_and_training_map_are_disjoint_and_frozen() -> None:
    manifest: Final = json.loads((ROOT / "heldout_tasks.json").read_text())
    score_payload: Final = json.loads((ROOT / "heldout_capability_map.json").read_text())
    heldout_ids: Final = {task["instance_id"] for task in manifest["tasks"]}
    training_ids: Final = set(score_payload["training_task_ids"])
    digest: Final = hashlib.sha256((ROOT / "heldout_capability_map.json").read_bytes()).hexdigest()
    expected_sha: Final = (ROOT / "heldout_capability_map.sha256").read_text().split()[0]

    assert heldout_ids.isdisjoint(training_ids)
    assert score_payload["training_only"] is True
    assert digest == expected_sha
    assert score_payload["scores"]["repository_debugging"]["haiku-4.5"]["score_usd_per_success"] == 0.73440015
    assert score_payload["scores"]["feature_implementation"]["haiku-4.5"]["score_usd_per_success"] is None


def test_baseline_tier_mapping_uses_active_tiers_and_defaults_unknown_to_sonnet() -> None:
    active: Final = active_tiers(source_config())

    assert [tier for tier, _ in active] == ["SIMPLE", "MEDIUM", "COMPLEX", "REASONING"]
    assert not any(tier == "NON_REASONING" for tier, _ in active)
    assert choose_baseline_route("SIMPLE", active).model == "haiku-4.5"
    assert choose_baseline_route("MEDIUM", active).model == "sonnet-5"
    assert choose_baseline_route("COMPLEX", active).model == "opus-5"
    assert choose_baseline_route("REASONING", active).model == "opus-5"
    assert choose_baseline_route("NON_REASONING", active).model == "sonnet-5"
    assert choose_baseline_route("OTHER", active).model == "sonnet-5"
    assert choose_baseline_route("SIMPLE", active, classifier_failed=True).reason == "classifier_failure"


def test_balanced_arm_order_is_deterministic_and_classifier_cost_is_separate() -> None:
    ids: Final = ("task-a", "task-b", "task-c", "task-d", "task-e")
    first: Final = balanced_schedule(ids)

    assert balanced_schedule(ids) == first
    assert sum(arms[0] == "capability" for _, arms in first) in {2, 3}
    assert classifier_plus_solver_cost(0.02, 0.5) == 0.52


def test_overlay_image_ids_are_separate_from_upstream_registry_digests() -> None:
    overlays: Final = _image_overlay_ids()
    provenance: Final = json.loads((ROOT / "heldout_image_provenance.json").read_text())

    assert overlays == {
        "swebench/sweb.eval.x86_64.pydicom_1776_pydicom-793":
            "sha256:cf79289615c0f96fbb827e8b4843d8d45f9a6c82fefcb3a2d982473dac1f6041",
        "swebench/sweb.eval.x86_64.pvlib_1776_pvlib-python-1191":
            "sha256:c963b822ba3abd17a92d0160257a4b4e32b3067014b609d36da15b107107e6f3",
    }
    assert all(
        item["upstream_registry_digest"] != item["local_overlay_image_id"]
        for item in provenance["images"]
        if item["local_overlay_image_id"] is not None
    )


def test_gateway_forwards_the_source_router_request_body_without_network() -> None:
    prompt: Final = "Sanitized problem statement only"
    source_capture: Final = CaptureClient()
    source, _ = source_router(source_capture)
    asyncio.run(source.aclassify(prompt=prompt))
    expected_body: Final = source_capture.request.model_dump(mode="json")
    forwarded: list[dict[str, object]] = []

    def transport(request: httpx.Request) -> httpx.Response:
        forwarded.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "answers": {
                    "tier": {
                        "type": "choice",
                        "choice": "SIMPLE",
                        "probabilities": {"SIMPLE": 1.0},
                        "confidence": 1.0,
                    }
                }
            },
            headers={"x-litellm-response-cost": "0.001", "x-litellm-call-id": "offline-test"},
            request=request,
        )

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
            gateway = GatewayJevClassifierClient(client, "https://gateway.invalid", "test-key")
            router, _ = source_router(gateway)
            await router.aclassify(prompt=prompt)

    asyncio.run(run())

    assert forwarded == [expected_body]
