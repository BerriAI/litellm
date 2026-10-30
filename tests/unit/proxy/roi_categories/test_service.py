from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Final

import pytest
from pydantic import ValidationError

from litellm.proxy.roi_categories.models import Assignment, Categories, Category, Settings, Traffic
from litellm.proxy.roi_categories.service import classify, discover
from litellm.router_strategy.complexity_router.jev_classifier import JevSystemOneRequest, JevSystemOneResponse

NOW: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)
SETTINGS: Final = Settings(
    version="test",
    started_at=NOW,
    sample_size=5,
    discovery_model="discovery",
    classifier_model="classifier",
    context="Support operations",
    categories=(Category(id="support", name="Support", description="Resolve customer tickets"),),
)


class Classifier:
    def __init__(self, choice: str = "support") -> None:
        self.choice: Final = choice
        self.requests: tuple[JevSystemOneRequest, ...] = ()

    async def evaluate(
        self, request: JevSystemOneRequest, timeout_s: float, request_kwargs: Mapping[str, object] | None = None
    ) -> JevSystemOneResponse:
        self.requests = (*self.requests, request)
        return JevSystemOneResponse.model_validate(
            {
                "answers": {
                    "category": {
                        "type": "choice",
                        "choice": self.choice,
                        "confidence": 0.9,
                        "probabilities": {self.choice: 0.9},
                    }
                }
            }
        )


@pytest.mark.asyncio
async def test_classification_uses_approved_criteria_and_preserves_request_identity() -> None:
    classifier: Final = Classifier()
    result: Final = await classify(
        SETTINGS, Traffic(request_id="request", excerpt="Help with my ticket", attempts=1), classifier
    )
    assert result == Assignment(request_id="request", version="test", category_id="support", attempts=2, completed=True)
    assert classifier.requests[0].questions["category"].criteria["support"] == "Resolve customer tickets"
    assert classifier.requests[0].state == "Help with my ticket"


@pytest.mark.asyncio
@pytest.mark.parametrize("choice", ["unclassified", "support"])
async def test_explicit_unclassified_and_approved_categories_are_terminal(choice: str) -> None:
    result: Final = await classify(SETTINGS, Traffic(request_id="request", excerpt="Question"), Classifier(choice))
    assert result.completed
    assert result.category_id == choice


@pytest.mark.asyncio
async def test_unknown_classifier_category_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown category"):
        await classify(SETTINGS, Traffic(request_id="request", excerpt="Question"), Classifier("invented"))


@pytest.mark.asyncio
async def test_discovery_waits_until_sample_is_full() -> None:
    async def unexpected_call(model: str, instructions: str, evidence: str) -> str:
        pytest.fail("Discovery must not call the model before the sample is full")

    result: Final = await discover(SETTINGS, (Traffic(request_id="one", excerpt="question"),), unexpected_call)
    assert result.phase == "sampling"
    assert result.sampled == 1


@pytest.mark.asyncio
async def test_discovery_includes_context_and_requires_review() -> None:
    async def complete(model: str, instructions: str, evidence: str) -> str:
        assert model == "discovery"
        assert "Support operations" in evidence
        assert "untrusted data" in instructions
        return '{"categories":[{"id":"support","name":"Support","description":"Resolve tickets"}]}'

    sample: Final = tuple(Traffic(request_id=str(index), excerpt="A customer ticket") for index in range(5))
    result: Final = await discover(SETTINGS, sample, complete)
    assert result.phase == "review"
    assert result.categories == (Category(id="support", name="Support", description="Resolve tickets"),)
    assert result.activated_at is None


@pytest.mark.parametrize("identifiers", [("same", "same"), ("unclassified",), ("bad'; DROP TABLE",)])
def test_category_ids_cannot_collide_or_use_reserved_names(identifiers: tuple[str, ...]) -> None:
    with pytest.raises(ValidationError):
        Categories.model_validate(
            {
                "categories": tuple(
                    {"id": identifier, "name": "Name", "description": "Purpose"} for identifier in identifiers
                )
            }
        )


def test_known_credentials_are_redacted_before_classification() -> None:
    credential: Final = "Bearer " + "a" * 40
    row: Final = Traffic(request_id="request", excerpt=f"Header {credential} while fixing a build")
    assert credential not in row.excerpt
    assert "fixing a build" in row.excerpt
