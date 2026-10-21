import hashlib

from poc.profile_manifest import (
    CandidateLabel,
    ImageAvailability,
    MAX_PER_CATEGORY,
    SEED,
    build_draft_manifest,
)


def test_manifest_selects_deterministic_public_images_and_marks_unsupported_categories() -> None:
    labels = (
        tuple(CandidateLabel(f"repo-{index}", "repository_debugging", None) for index in range(35))
        + tuple(CandidateLabel(f"feature-{index}", "feature_implementation", None) for index in range(32))
        + (
            CandidateLabel("shell-1", "shell_build", None),
            CandidateLabel("unknown-1", "unknown", None),
        )
    )
    images = tuple(
        ImageAvailability(
            label.task_id,
            f"image/{label.task_id}",
            True,
            f"sha256:{label.task_id}",
            100,
            None,
            True,
        )
        for label in labels
    )
    manifest = build_draft_manifest(
        tuple(reversed(labels)),
        tuple(reversed(images)),
        frozen_at_utc="2026-10-21T00:00:00+00:00",
    )
    categories = manifest["categories"]
    repo_expected = tuple(
        sorted(
            (label.task_id for label in labels if label.primary_category == "repository_debugging"),
            key=lambda task_id: (hashlib.sha256(f"{SEED}{task_id}".encode()).hexdigest(), task_id),
        )[:MAX_PER_CATEGORY]
    )
    feature_expected = tuple(
        sorted(
            (label.task_id for label in labels if label.primary_category == "feature_implementation"),
            key=lambda task_id: (hashlib.sha256(f"{SEED}{task_id}".encode()).hexdigest(), task_id),
        )[:MAX_PER_CATEGORY]
    )
    assert manifest["frozen"] is True
    assert manifest["frozen_at_utc"] == "2026-10-21T00:00:00+00:00"
    assert categories["repository_debugging"]["selected_ids"] == repo_expected
    assert categories["feature_implementation"]["selected_ids"] == feature_expected
    assert categories["shell_build"]["selection_status"] == "unsupported_category"
    assert categories["shell_build"]["selected"] == 0
    assert categories["algorithms_data"]["selected"] == 0
    assert manifest["unknown_count"] == 1


def test_manifest_reports_image_unavailable_and_lookup_errors_without_selecting_them() -> None:
    labels = (
        CandidateLabel("available", "repository_debugging", None),
        CandidateLabel("unavailable", "repository_debugging", None),
        CandidateLabel("error", "repository_debugging", None),
        CandidateLabel("not-inspected", "repository_debugging", None),
    )
    images = (
        ImageAvailability("available", "image/a", True, "sha256:a", 100, None, True),
        ImageAvailability("unavailable", "image/b", False, None, None, None, True),
        ImageAvailability(
            "error",
            "image/c",
            None,
            None,
            None,
            "access_error",
            True,
            token_scope_ok=False,
            retry_status=401,
            auth_rate_limit_remaining="17",
            auth_retry_after="2",
            rate_limit_remaining="41",
            retry_after="3",
        ),
    )
    image_by_id = {image.task_id: image for image in images}
    manifest = build_draft_manifest(labels, images)
    counts = manifest["categories"]["repository_debugging"]
    assert counts["labeled"] == 4
    assert counts["inspected"] == 3
    assert counts["with_image"] == 1
    assert counts["image_unavailable"] == 1
    assert counts["image_lookup_errors"] == 1
    assert counts["not_inspected"] == 1
    ordered_ids = tuple(
        sorted(
            (label.task_id for label in labels),
            key=lambda task_id: (
                hashlib.sha256(f"{SEED}{task_id}".encode()).hexdigest(),
                task_id,
            ),
        )
    )
    assert counts["selected_ids"] == tuple(
        task_id
        for task_id in ordered_ids
        if image_by_id.get(task_id) is not None and image_by_id[task_id].exists is True
    )
    assert manifest["image_unavailable_count"] == 1
    assert manifest["image_lookup_error_count"] == 1
    assert manifest["error_exclusions"] == (
        {
            "task_id": "error",
            "category": "repository_debugging",
            "status": "access_error",
            "reason": "access_error",
            "retry_status": 401,
            "token_scope_ok": False,
            "auth_rate_limit_remaining": "17",
            "auth_retry_after": "2",
            "rate_limit_remaining": "41",
            "retry_after": "3",
        },
    )
