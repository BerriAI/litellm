import calendar
import re
from collections.abc import Mapping
from datetime import date, datetime, timedelta, timezone
from itertools import chain
from statistics import median
from types import MappingProxyType
from typing import Final

from litellm.proxy.roi_calculator.analytics import normalize_email
from litellm.proxy.roi_calculator.branch_spend import attribute_branch_keys
from litellm.types.roi_observed import (
    ObservedAccount,
    ObservedData,
    ObservedHumanSummary,
    ObservedPeriod,
    ObservedPeriodData,
    ObservedPeriods,
    ObservedPerson,
    ObservedPersonPeriod,
    ObservedPersonPeriods,
    ObservedPull,
    ObservedPullPeriods,
    ObservedPullResponse,
    ObservedReport,
    ObservedWindow,
)


def reporting_windows(now: datetime, days: int = 28) -> tuple[ObservedWindow, ObservedWindow, ObservedWindow]:
    end: Final = now.astimezone(timezone.utc).date() - timedelta(days=1)
    start: Final = end - timedelta(days=days - 1)
    last_year_end: Final = date(end.year - 1, end.month, min(end.day, calendar.monthrange(end.year - 1, end.month)[1]))
    return (
        ObservedWindow(start=start, end=end),
        ObservedWindow(start=start - timedelta(days=days), end=start - timedelta(days=1)),
        ObservedWindow(start=last_year_end - timedelta(days=days - 1), end=last_year_end),
    )


def declared_requester(author: str, body: str) -> str:
    if author.casefold().removesuffix("[bot]") not in ("devin-ai-integration", "devin-ai"):
        return ""
    matches: Final = frozenset(
        match.group(1) for match in re.finditer(r"^Requested by:\s*@([A-Za-z0-9_.-]+)\s*$", body, re.MULTILINE)
    )
    return next(iter(matches)).casefold() if len(matches) == 1 else ""


def merge_hours(pull: ObservedPull) -> float | None:
    if pull.created_at is None or pull.created_at.tzinfo is None or pull.merged_at.tzinfo is None:
        return None
    seconds: Final = (pull.merged_at - pull.created_at).total_seconds()
    return seconds / 3600 if seconds >= 0 else None


def median_hours(pulls: tuple[ObservedPull, ...]) -> float | None:
    values: Final = tuple(hours for pull in pulls if (hours := merge_hours(pull)) is not None)
    return median(values) if values else None


def _owner_login(pull: ObservedPull) -> str:
    login: Final = (pull.requester if pull.agent else pull.author).casefold()
    return f"{pull.connection_id}:{login}" if pull.connection_id and login else login


def identity_matches(data: ObservedData, manual: Mapping[str, str], ignored: tuple[str, ...] = ()) -> Mapping[str, str]:
    pulls: Final = tuple(chain(data.current.pulls, data.previous.pulls, data.last_year.pulls))
    candidates: Final = frozenset((_owner_login(pull), normalize_email(pull.profile_email)) for pull in pulls)
    gateway_emails: Final = frozenset(data.gateway_emails)
    profiles: Final = {
        login: email
        for login, email in candidates
        if login not in ignored
        and email in gateway_emails
        and len({value for key, value in candidates if key == login and value}) == 1
    }
    return MappingProxyType({**profiles, **manual})


def _person_period(data: ObservedPeriodData, email: str, identities: Mapping[str, str]) -> ObservedPersonPeriod:
    pulls: Final = tuple(pull for pull in data.pulls if identities.get(_owner_login(pull)) == email)
    spend: Final = tuple(row for row in data.spend if row["email"] == email)
    cost: Final = sum(row["spend"] for row in spend)
    days: Final = (data.window.end - data.window.start).days + 1
    return ObservedPersonPeriod(
        merged_prs=len(pulls),
        prs_per_week=len(pulls) * 7 / days,
        median_merge_hours=median_hours(pulls),
        direct_authored=sum(not pull.agent for pull in pulls),
        declared_agent_owned=sum(pull.agent for pull in pulls),
        gateway_recorded_spend=cost,
        recorded_spend_per_attributed_pr=cost / len(pulls) if spend and pulls else None,
        spend_observation="records_present" if spend else "no_records",
        pr_urls=tuple(pull.url for pull in pulls),
    )


def _person(data: ObservedData, email: str, identities: Mapping[str, str]) -> ObservedPerson:
    return ObservedPerson(
        name=email.split("@", 1)[0],
        email=email,
        logins=tuple(
            sorted(frozenset(login.rsplit(":", 1)[-1] for login, address in identities.items() if address == email))
        ),
        accounts=tuple(
            ObservedAccount(
                connection_id=login.split(":", 1)[0] if ":" in login else "", login=login.rsplit(":", 1)[-1]
            )
            for login, address in identities.items()
            if address == email
        ),
        periods=ObservedPersonPeriods(
            current=_person_period(data.current, email, identities),
            previous=_person_period(data.previous, email, identities),
            last_year=_person_period(data.last_year, email, identities),
        ),
    )


def _issue_count(data: ObservedPeriodData, labels: frozenset[str]) -> int | None:
    if data.issues is None:
        return None
    return sum(bool(labels.intersection(label.casefold().strip() for label in issue.labels)) for issue in data.issues)


def _period(data: ObservedPeriodData, identities: Mapping[str, str]) -> ObservedPeriod:
    matched: Final = tuple(pull for pull in data.pulls if _owner_login(pull) in identities)
    emails: Final = frozenset(identities.values())
    spend: Final = tuple(row for row in data.spend if row["email"] in emails)
    humans: Final = tuple(pull for pull in data.pulls if pull.author and not pull.agent)
    return ObservedPeriod(
        window=data.window,
        merged_prs=len(data.pulls),
        median_merge_hours=median_hours(data.pulls),
        human_authored=len(humans),
        agent_authored=sum(pull.agent for pull in data.pulls),
        missing_author=sum(not pull.author for pull in data.pulls),
        agents_without_requester=sum(pull.agent and not pull.requester for pull in data.pulls),
        matched_internal_prs=len(matched),
        new_bug_labeled_issues=_issue_count(data, frozenset(("bug", "kind:bug", "type::bug"))),
        new_regression_labeled_issues=_issue_count(
            data, frozenset(("regression", "kind:regression", "type::regression"))
        ),
        explicitly_titled_revert_prs=sum(
            bool(re.match(r"^revert(?:\W|$)", pull.title, re.IGNORECASE)) for pull in data.pulls
        ),
        matched_users_recorded_spend=sum(row["spend"] for row in spend),
        spend_observation="records_present" if spend else "no_records",
        human_summary=ObservedHumanSummary(median_merge_hours=median_hours(humans)),
    )


def _pulls(data: ObservedPeriodData) -> tuple[ObservedPullResponse, ...]:
    costs: Final = attribute_branch_keys(
        tuple((pull.url, pull.number, pull.source_repo, pull.source_branch) for pull in data.pulls), data.branch_spend
    )
    return tuple(
        ObservedPullResponse.model_validate(
            {**pull.model_dump(), "merge_hours": merge_hours(pull), "branch_cost": costs[(pull.url, pull.number)]}
        )
        for pull in data.pulls
    )


def summarize_observed(data: ObservedData, manual: Mapping[str, str], ignored: tuple[str, ...] = ()) -> ObservedReport:
    identities: Final = identity_matches(data, manual, ignored)
    all_pulls: Final = tuple(chain(data.current.pulls, data.previous.pulls, data.last_year.pulls))
    current_pulls: Final = _pulls(data.current)
    linked_branches: Final = frozenset(
        (pull.source_repo, pull.source_branch) for pull in current_pulls if pull.branch_cost.status == "matched"
    )
    return ObservedReport(
        source_provider=data.source_provider,
        connections=data.connections,
        repos=data.repos,
        captured_at=data.captured_at,
        periods=ObservedPeriods(
            current=_period(data.current, identities),
            previous=_period(data.previous, identities),
            last_year=_period(data.last_year, identities),
        ),
        people=tuple(_person(data, email, identities) for email in sorted(frozenset(identities.values()))),
        pulls=ObservedPullPeriods(
            current=current_pulls, previous=_pulls(data.previous), last_year=_pulls(data.last_year)
        ),
        unlinked_branches=tuple(
            row for row in data.current.branch_spend or () if (row.repo, row.branch) not in linked_branches
        ),
        unmatched_logins=tuple(sorted(frozenset(_owner_login(pull) for pull in all_pulls) - identities.keys() - {""})),
    )
