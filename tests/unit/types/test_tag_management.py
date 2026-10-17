from litellm.types.tag_management import TagConfig, TagUpdateRequest


def test_tag_update_request_accepts_spend():
    request = TagUpdateRequest(name="batch-jobs", spend=0.0)
    assert request.spend == 0.0
    assert "spend" in request.model_fields_set


def test_tag_update_request_spend_defaults_to_none():
    request = TagUpdateRequest(name="batch-jobs")
    assert request.spend is None
    assert "spend" not in request.model_fields_set


def test_tag_config_accepts_spend():
    config = TagConfig(
        name="batch-jobs",
        created_at="2026-01-01T00:00:00",
        updated_at="2026-01-01T00:00:00",
        spend=42.5,
    )
    assert config.spend == 42.5


def test_tag_config_spend_defaults_to_none():
    config = TagConfig(
        name="batch-jobs",
        created_at="2026-01-01T00:00:00",
        updated_at="2026-01-01T00:00:00",
    )
    assert config.spend is None
