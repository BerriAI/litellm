from typing import Final

from litellm.proxy.lens.models import Check
from litellm.proxy.lens.reviews import criteria_key
from tests.unit.proxy.lens.test_state import lens


def test_only_evaluation_changes_invalidate_reviews() -> None:
    settings: Final = lens().settings
    operations: Final = settings.model_copy(
        update={
            "name": "Renamed",
            "monthly_budget": 200,
            "interval_minutes": 30,
            "enabled": False,
            "concurrency": 2,
        }
    )
    assert criteria_key(operations) == criteria_key(settings)
    assert criteria_key(settings.model_copy(update={"context": "Only inspect unrecovered errors"})) != criteria_key(
        settings
    )
    assert criteria_key(
        settings.model_copy(update={"checks": (Check(id="retries", instruction="Find all retries"),)})
    ) != criteria_key(settings)
    assert criteria_key(settings.model_copy(update={"model": "another-analysis-model"})) != criteria_key(settings)
