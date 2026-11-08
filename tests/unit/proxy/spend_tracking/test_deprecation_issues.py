from datetime import date, datetime, timedelta, timezone
from typing import Final

import pytest

from litellm.proxy.spend_tracking.deprecation_issues import deprecation_candidates, used_deprecation_issues
from litellm.types.proxy.usage_alerts import UsageAlertMetricPoint
from litellm.types.proxy.model_deprecation import (
    DeprecationAlertState,
    DeprecationAlertStatus,
    DeprecationStatus,
    ModelDeprecationInfo,
    ModelDeprecationUsage,
    SuccessorModelSource,
)

NOW = datetime(2026, 11, 5, 10, tzinfo=timezone.utc)
RETIREMENT = date(2026, 11, 20)


def model(
    *,
    model_name: str = "public-alias",
    litellm_model: str = "example/retiring-model",
    deprecation_date: date = RETIREMENT,
    days_until_deprecation: int = 15,
    status: DeprecationStatus = "imminent",
    successor_model: str | None = None,
    successor_model_source: SuccessorModelSource | None = None,
) -> ModelDeprecationInfo:
    return ModelDeprecationInfo(
        model_name=model_name,
        litellm_model=litellm_model,
        litellm_provider="example",
        deprecation_date=deprecation_date,
        days_until_deprecation=days_until_deprecation,
        status=status,
        successor_model=successor_model,
        successor_model_source=successor_model_source,
    )


def snapshot(
    *models: ModelDeprecationInfo, upcoming: tuple[ModelDeprecationInfo, ...] = ()
) -> tuple[ModelDeprecationInfo, ...]:
    return (*models, *upcoming)


def usage(name: str = "example/retiring-model") -> ModelDeprecationUsage:
    return ModelDeprecationUsage(
        model=name,
        requests=700,
        last_used_date=NOW.date(),
        last_day_requests=30,
        keys=12,
        users=4,
        teams=2,
    )


def test_aliases_share_one_issue_only_when_actual_retiring_model_has_usage():
    candidates = deprecation_candidates(snapshot(model(), model(model_name="second-alias")))
    assert used_deprecation_issues(candidates, {"public-alias": usage("public-alias")}, {}, NOW) == ()
    issues = used_deprecation_issues(candidates, {usage().model: usage()}, {}, NOW)
    assert len(issues) == 1
    issue = issues[0]
    assert issue.cohort.scope_id == "example/retiring-model"
    assert issue.impact.keys == 12
    assert issue.impact.requests == 700
    assert issue.range_start == date(2026, 10, 7)
    assert issue.range_end == NOW.date()


def test_retirement_identity_survives_urgency_and_recommendation_changes():
    before = deprecation_candidates(snapshot(model()))
    after = deprecation_candidates(
        snapshot(
            model(
                status="deprecated",
                days_until_deprecation=-1,
                successor_model="example/replacement",
                successor_model_source=SuccessorModelSource.DEPLOYMENT_CONFIG,
            )
        )
    )
    original = used_deprecation_issues(before, {usage().model: usage()}, {}, NOW)[0]
    later = datetime(2026, 11, 21, tzinfo=timezone.utc)
    changed = used_deprecation_issues(after, {usage().model: usage()}, {}, later)[0]
    assert original.id == changed.id
    assert original.cohort_key == changed.cohort_key
    assert changed.title.endswith("retired on 2026-11-20")
    assert changed.range_end == later.date()
    assert changed.deprecation.model.successor_model == "example/replacement"
    assert changed.deprecation.model.successor_model_source == SuccessorModelSource.DEPLOYMENT_CONFIG


def test_each_retirement_date_keeps_its_existing_acknowledgment_identity():
    candidates = deprecation_candidates(snapshot(model(), model(deprecation_date=RETIREMENT + timedelta(days=1))))
    first = candidates[0]
    state = DeprecationAlertState(
        model=first.usage_model,
        deprecation_date=first.model.deprecation_date,
        status=DeprecationAlertStatus.DISMISSED,
        last_seen_date=NOW.date(),
        last_seen_requests=30,
    )
    issues = used_deprecation_issues(candidates, {usage().model: usage()}, {first.state_key: state}, NOW)
    assert len({issue.id for issue in issues}) == 2
    assert issues[0].deprecation.state == state
    assert issues[1].deprecation.state.status == DeprecationAlertStatus.OPEN


def test_provider_prefix_candidates_match_only_exact_recorded_names():
    candidates = deprecation_candidates(snapshot(model(litellm_model="retiring-model")))
    rows = tuple(usage(name) for name in ("retiring-model", "example/retiring-model", "other/retiring-model"))
    issues = used_deprecation_issues(candidates, {row.model: row for row in rows}, {}, NOW)
    assert {issue.cohort.scope_id for issue in issues} == {"retiring-model", "example/retiring-model"}


def test_unused_and_future_retirements_do_not_create_issues():
    candidates = deprecation_candidates(
        snapshot(model(litellm_model="example/unused"), upcoming=(model(status="upcoming", days_until_deprecation=60),))
    )
    assert used_deprecation_issues(candidates, {usage().model: usage()}, {}, NOW) == ()


def test_maximum_model_identity_is_preserved_in_activity_and_evidence():
    name = "example/" + "m" * (1024 - len("example/"))
    candidates = deprecation_candidates(snapshot(model(litellm_model=name)))
    issue = used_deprecation_issues(candidates, {name: usage(name)}, {}, NOW)[0]
    assert issue.cohort.scope_id == name
    assert issue.deprecation.usage_model == name
    assert issue.title.startswith("public-alias")


def test_retirement_candidates_are_bounded_and_preserve_alias_choice() -> None:
    candidates = deprecation_candidates((model(model_name="z-alias"), model(model_name="a-alias")))
    assert len(candidates) == 1 and candidates[0].model.model_name == "a-alias"
    with pytest.raises(ValueError, match="at most 10"):
        deprecation_candidates(tuple(model(model_name=f"alias-{index}") for index in range(11)))


def test_retirement_keeps_recorded_daily_usage_in_the_saved_issue() -> None:
    points: Final = (
        UsageAlertMetricPoint(day=NOW.date() - timedelta(days=2), value=670),
        UsageAlertMetricPoint(day=NOW.date(), value=30),
    )
    recorded: Final = usage().model_copy(update={"daily_requests": points})
    issue = used_deprecation_issues(deprecation_candidates((model(),)), {recorded.model: recorded}, {}, NOW)[0]
    assert issue.deprecation is not None
    assert issue.deprecation.daily_requests == points
    assert sum(point.value or 0 for point in issue.deprecation.daily_requests) == issue.deprecation.requests
