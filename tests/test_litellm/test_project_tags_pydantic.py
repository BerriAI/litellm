import pytest
from litellm.proxy._types import NewProjectRequest, UpdateProjectRequest


def test_new_project_request_tags():
    # Test tags correctly stay top level initially
    req = NewProjectRequest(
        project_id="test_proj", team_id="team_1", tags=["tag1", "tag2"]
    )

    # tags should be top level initially
    assert req.tags == ["tag1", "tag2"]


def test_update_project_request_tags():
    # Test tags correctly stay top level initially
    req = UpdateProjectRequest(project_id="test_proj", tags=["new_tag"])

    assert req.tags == ["new_tag"]


def test_new_project_request_invalid_tags_type():
    # tags must be a list — a string should raise a ValidationError
    with pytest.raises(Exception):
        NewProjectRequest(project_id="test_proj", team_id="team_1", tags="not-a-list")


def test_update_project_request_invalid_tags_type():
    # tags must be a list — a string should raise a ValidationError
    with pytest.raises(Exception):
        UpdateProjectRequest(project_id="test_proj", tags="not-a-list")
