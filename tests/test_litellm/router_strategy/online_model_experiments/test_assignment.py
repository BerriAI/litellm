from litellm.router_strategy.online_model_experiments.assignment import (
    ExperimentAssignment,
    ExperimentAssignmentError,
    ExperimentVariant,
    assign_variant,
    derive_assignment_key,
)


def test_assignment_is_stable_for_same_experiment_and_identity():
    assignment_key = derive_assignment_key("support", "user-123", "test-secret")
    assert isinstance(assignment_key, str)
    variants = (
        ExperimentVariant(name="model-a", weight_basis_points=5000),
        ExperimentVariant(name="model-b", weight_basis_points=5000),
    )

    first = assign_variant("support", assignment_key, variants)
    second = assign_variant("support", assignment_key, variants)

    assert isinstance(first, ExperimentAssignment)
    assert second == first


def test_assignment_key_does_not_expose_identity():
    assignment_key = derive_assignment_key("support", "user-123", "test-secret")

    assert isinstance(assignment_key, str)
    assert assignment_key != "user-123"
    assert "user-123" not in assignment_key


def test_assignment_changes_when_experiment_changes():
    first = derive_assignment_key("support-v1", "user-123", "test-secret")
    second = derive_assignment_key("support-v2", "user-123", "test-secret")

    assert isinstance(first, str)
    assert isinstance(second, str)
    assert first != second


def test_assignment_rejects_invalid_allocation():
    result = assign_variant(
        "support",
        "assignment-key",
        (ExperimentVariant(name="model-a", weight_basis_points=7000),),
    )

    assert result == ExperimentAssignmentError(
        code="invalid_weights",
        message="variant weights must sum to 10000 basis points",
    )


def test_assignment_rejects_duplicate_variant_names():
    result = assign_variant(
        "support",
        "assignment-key",
        (
            ExperimentVariant(name="model-a", weight_basis_points=5000),
            ExperimentVariant(name="model-a", weight_basis_points=5000),
        ),
    )

    assert result == ExperimentAssignmentError(
        code="invalid_variants",
        message="variants must be non-empty and uniquely named",
    )


def test_key_derivation_reports_missing_secret():
    result = derive_assignment_key("support", "user-123", "")

    assert result == ExperimentAssignmentError(
        code="empty_secret",
        message="secret must not be empty",
    )
