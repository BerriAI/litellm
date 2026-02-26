import pytest
from litellm.proxy._types import NewProjectRequest, UpdateProjectRequest


def test_new_project_request_tags():
    # Test tags are correctly moved to metadata["tags"]
    req = NewProjectRequest(
        project_id="test_proj", team_id="team_1", tags=["tag1", "tag2"]
    )

    # After validation, tags should be inside metadata
    assert req.metadata is not None
    assert "tags" in req.metadata
    assert req.metadata["tags"] == ["tag1", "tag2"]
    assert req.tags is None  # Or removed dependending on pydantic version


def test_update_project_request_tags():
    # Test tags are correctly moved to metadata["tags"]
    req = UpdateProjectRequest(project_id="test_proj", tags=["new_tag"])

    assert req.metadata is not None
    assert "tags" in req.metadata
    assert req.metadata["tags"] == ["new_tag"]
    assert req.tags is None


def test_new_project_request_invalid_tags_type():
    # tags must be a list — a string should raise a ValidationError
    with pytest.raises(Exception):
        NewProjectRequest(project_id="test_proj", team_id="team_1", tags="not-a-list")


def test_update_project_request_invalid_tags_type():
    # tags must be a list — a string should raise a ValidationError
    with pytest.raises(Exception):
        UpdateProjectRequest(project_id="test_proj", tags="not-a-list")
