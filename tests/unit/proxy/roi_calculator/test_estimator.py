from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import pytest
from pydantic import TypeAdapter

from litellm.proxy.roi_calculator.estimator import Estimator
from litellm.proxy.roi_calculator.github import SourceError
from litellm.types.roi_calculator import (
    ROICompletionRequest,
    ROIEstimatorChanges,
    ROIEstimatorEvidence,
    ROIPullEvidence,
    ROIResponseFormat,
    ROISettings,
)


def _pull() -> ROIPullEvidence:
    pull: Final[ROIPullEvidence] = {
        "repo": "org/repo",
        "number": 42,
        "title": "Fix timezone conversion",
        "body": "Preserve UTC behavior.",
        "url": "https://github.com/org/repo/pull/42",
        "login": "alice",
        "emails": ("alice@example.com",),
        "profile_email": "alice@example.com",
        "merged_at": "2026-09-12T12:00:00Z",
        "head_sha": "abcdef",
        "additions": 1,
        "deletions": 1,
        "changed_files": 1,
        "files": (
            {"filename": "time.py", "status": "modified", "additions": 1, "deletions": 1},
        ),
        "commits": ({"sha": "abcdef", "message": "Fix timezone conversion"},),
        "commit_count": 1,
        "incomplete_metadata": False,
    }
    return pull


def _settings() -> ROISettings:
    return ROISettings(estimator_model="test-estimator")


def _completion(content: str) -> Mapping[str, object]:
    message: Final = MappingProxyType({"content": content})
    choice: Final = MappingProxyType({"finish_reason": "stop", "message": message})
    response: Final = MappingProxyType({"choices": (choice,)})
    return response


@pytest.mark.asyncio
async def test_estimator_sends_metadata_only_json_request_and_parses_valid_result() -> None:
    async def complete(request: ROICompletionRequest) -> object:
        evidence: Final = TypeAdapter(ROIEstimatorEvidence).validate_json(request.messages[1]["content"])
        assert request.temperature == 0
        expected_response_format: Final[ROIResponseFormat] = {"type": "json_object"}
        assert request.response_format == expected_response_format
        assert "patch" not in request.messages[1]["content"]
        assert "alice@example.com" not in request.messages[1]["content"]
        expected_changes: Final = ROIEstimatorChanges(additions=1, deletions=1, files=1, commits=1)
        assert evidence.changes == expected_changes
        assert evidence.commits[0].message == "Fix timezone conversion"
        assert "without AI assistance" in request.messages[0]["content"]
        return _completion('{"hours": 4.25, "reasoning": "Timezone conversion and regression verification."}')

    result: Final = await Estimator(_settings(), complete).estimate(_pull())

    assert result["hours"] == 4.25
    assert result.get("effort_basis") == "without_ai"


@pytest.mark.parametrize(
    "content",
    (
        '{"hours": -1, "reasoning": "invalid"}',
        '{"hours": NaN, "reasoning": "invalid"}',
        '{"hours": "4", "reasoning": "invalid"}',
        '{"hours": true, "reasoning": "invalid"}',
        '{"hours": 4}',
        '{"hours": 4, "reasoning": "  "}',
        "not json",
    ),
)
@pytest.mark.asyncio
async def test_estimator_rejects_invalid_hours_or_reasoning(content: str) -> None:
    async def complete(request: ROICompletionRequest) -> object:
        return _completion(content)

    with pytest.raises(SourceError):
        await Estimator(_settings(), complete).estimate(_pull())


@pytest.mark.asyncio
async def test_incomplete_metadata_is_not_sent_to_the_estimator() -> None:
    async def complete(request: ROICompletionRequest) -> object:
        raise AssertionError("Incomplete metadata must not reach the estimator.")

    pull: Final[ROIPullEvidence] = {**_pull(), "incomplete_metadata": True}

    result: Final = await Estimator(_settings(), complete).estimate(pull)

    assert result["status"] == "needs_review"
