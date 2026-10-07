import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.lens.feedback import FeedbackCreate, FeedbackUpdate, author


@pytest.mark.parametrize("value", range(11))
def test_usefulness_accepts_every_integer_including_zero(value: int) -> None:
    assert FeedbackCreate(trace_id="trace", value=value).value == value
    assert FeedbackUpdate(value=value).value == value


@pytest.mark.parametrize("value", [-1, 11, 3.5, 8.0, True, False, "8", None])
def test_usefulness_rejects_out_of_range_and_coerced_values(value: object) -> None:
    with pytest.raises(ValidationError):
        FeedbackCreate.model_validate({"trace_id": "trace", "value": value})
    with pytest.raises(ValidationError):
        FeedbackUpdate.model_validate({"value": value})


@pytest.mark.parametrize("extra", [{"key": "other"}, {"author_id": "victim"}, {"comment": "x" * 4001}])
def test_feedback_cannot_override_criterion_or_author(extra: dict[str, str]) -> None:
    with pytest.raises(ValidationError):
        FeedbackCreate.model_validate({"trace_id": "trace", "value": 8, **extra})


@pytest.mark.parametrize("role", [LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY, LitellmUserRoles.INTERNAL_USER_VIEW_ONLY])
def test_read_only_reviewers_cannot_write(role: LitellmUserRoles) -> None:
    auth = UserAPIKeyAuth(user_id="reader", user_role=role)
    with pytest.raises(HTTPException) as error:
        author(auth, write=True)
    assert error.value.status_code == 403


def test_identity_survives_key_rotation_and_separates_reviewers() -> None:
    first = author(UserAPIKeyAuth(user_id="alice", token="key-1"))
    assert first == author(UserAPIKeyAuth(user_id="alice", token="key-2"))
    assert first != author(UserAPIKeyAuth(user_id="bob", token="key-1"))
    assert first != "alice"


def test_anonymous_requests_cannot_become_a_shared_reviewer() -> None:
    with pytest.raises(HTTPException) as error:
        author(UserAPIKeyAuth())
    assert error.value.status_code == 403
