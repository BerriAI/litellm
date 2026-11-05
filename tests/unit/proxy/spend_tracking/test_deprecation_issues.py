from datetime import date, datetime, timedelta, timezone

from litellm.proxy.spend_tracking.deprecation_issues import deprecation_candidates, used_deprecation_issues
from litellm.types.proxy.model_deprecation import (
    DeprecationAlertState,
    DeprecationAlertStatus,
    ModelDeprecationInfo,
    ModelDeprecationResponse,
    ModelDeprecationUsage,
    SuccessorModelSource,
)

NOW = datetime(2026, 11, 5, 10, tzinfo=timezone.utc)
RETIREMENT = date(2026, 11, 20)


def model(**updates):
    return ModelDeprecationInfo(
        **{
            "model_name": "public-alias",
            "litellm_model": "example/retiring-model",
            "litellm_provider": "example",
            "deprecation_date": RETIREMENT,
            "days_until_deprecation": 15,
            "status": "imminent",
            **updates,
        }
    )


def snapshot(*models, upcoming=()):
    return ModelDeprecationResponse(
        imminent=[entry for entry in models if entry.status == "imminent"],
        deprecated=[entry for entry in models if entry.status == "deprecated"],
        upcoming=upcoming,
        warn_within_days=30,
        checked_at=NOW,
    )


def usage(name="example/retiring-model", **updates):
    return ModelDeprecationUsage(
        **{
            "model": name,
            "requests": 700,
            "last_used_date": NOW.date(),
            "last_day_requests": 30,
            "keys": 12,
            "users": 4,
            "teams": 2,
            **updates,
        }
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
