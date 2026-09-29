import re
from collections.abc import Mapping
from typing import Final

from litellm.types.roi_calculator import (
    ROIPersonSummary,
    ROIPullRecord,
    ROIPullSummary,
    ROIReport,
    ROISpendRecord,
    ROISummary,
    ROISummaryMetrics,
    ROITrendDay,
)

_EMAIL_PATTERN: Final = re.compile(r"[^\s@]+@[^\s@]+\.[^\s@]+")


def normalize_email(value: str | None) -> str:
    normalized: Final = (value or "").strip().casefold()
    if _EMAIL_PATTERN.fullmatch(normalized) is None or normalized.endswith("noreply.github.com"):
        return ""
    return normalized


def match_identity(
    pull: ROIPullRecord,
    observed_emails: frozenset[str],
    mappings: Mapping[str, str],
) -> tuple[str, str]:
    mapped: Final = mappings.get(pull["login"].casefold())
    if mapped:
        return normalize_email(mapped), "manual"
    candidates: Final = frozenset(
        address
        for address in (normalize_email(candidate) for candidate in pull["emails"])
        if address
    )
    matched: Final = candidates & observed_emails
    if len(matched) == 1:
        address: Final = next(iter(matched))
        return address, "profile email" if address == normalize_email(pull["profile_email"]) else "commit email"
    if len(matched) > 1:
        return "", "ambiguous emails"
    return "", "email unavailable" if not candidates else "no gateway match"


def _person_key(address: str, fallback: str) -> str:
    return address or fallback


def _pull_summary(
    pull: ROIPullRecord,
    address: str,
    method: str,
    observed: frozenset[str],
) -> ROIPullSummary:
    return ROIPullSummary(
        repo=pull["repo"],
        number=pull["number"],
        title=pull["title"],
        url=pull["url"],
        login=pull["login"],
        emails=pull["emails"],
        profile_email=pull["profile_email"],
        merged_at=pull["merged_at"],
        head_sha=pull["head_sha"],
        additions=pull["additions"],
        deletions=pull["deletions"],
        changed_files=pull["changed_files"],
        commit_count=pull["commit_count"],
        incomplete_metadata=pull["incomplete_metadata"],
        estimate=pull["estimate"],
        cache_key=pull.get("cache_key"),
        email=address,
        match_method=method,
        matched=address in observed,
    )


def _summarize_person(
    key: str,
    spend: tuple[ROISpendRecord, ...],
    pulls: tuple[tuple[ROIPullRecord, str, str], ...],
) -> ROIPersonSummary:
    spend_rows: Final = tuple(
        row
        for row in spend
        if _person_key(normalize_email(row["email"]), "gateway:" + row["user_id"]) == key
    )
    person_pulls: Final = tuple(
        pull
        for pull in pulls
        if _person_key(pull[1], "github:" + pull[0]["login"].casefold()) == key
    )
    addresses: Final = tuple(normalize_email(row["email"]) for row in spend_rows if row["email"])
    person_email: Final = addresses[0] if addresses else (person_pulls[0][1] if person_pulls else "")
    spend_total: Final[float | None] = sum(row["spend"] for row in spend_rows) if spend_rows else None
    login_values: Final = tuple(pull[0]["login"] for pull in person_pulls)
    logins: Final = tuple(
        login for index, login in enumerate(login_values) if login not in login_values[:index]
    )
    method_values: Final = tuple(pull[2] for pull in person_pulls)
    methods: Final = tuple(
        method for index, method in enumerate(method_values) if method not in method_values[:index]
    )
    estimates: Final = tuple(pull[0]["estimate"] for pull in person_pulls)
    estimated_count: Final = sum(estimate["status"] == "estimated" for estimate in estimates)
    pending_count: Final = len(estimates) - estimated_count
    hours: Final = sum(
        estimate["hours"] or 0.0 for estimate in estimates if estimate["status"] == "estimated"
    )
    eligible: Final = spend_total is not None and estimated_count > 0 and pending_count == 0
    return ROIPersonSummary(
        id=key,
        email=person_email,
        logins=logins,
        spend=spend_total,
        hours=hours,
        prs=len(person_pulls),
        estimated_prs=estimated_count,
        pending_prs=pending_count,
        match_methods=methods,
        eligible=eligible,
        cost_per_hour=spend_total / hours if eligible and hours > 0 and spend_total is not None else None,
    )


def summarize(report: ROIReport, mappings: Mapping[str, str]) -> ROISummary:
    observed: Final = frozenset(
        normalized
        for normalized in (normalize_email(row["email"]) for row in report["spend"])
        if normalized
    )
    matched_pulls: Final[tuple[tuple[ROIPullRecord, str, str], ...]] = tuple(
        (pull, *match_identity(pull, observed, mappings)) for pull in report["pulls"]
    )
    gateway_people: Final = frozenset(
        _person_key(normalize_email(row["email"]), "gateway:" + row["user_id"])
        for row in report["spend"]
    )
    github_people: Final = frozenset(
        _person_key(address, "github:" + pull["login"].casefold())
        for pull, address, _ in matched_pulls
    )
    people_keys: Final = gateway_people | github_people
    people: Final = tuple(
        _summarize_person(
            key,
            report["spend"],
            matched_pulls,
        )
        for key in sorted(people_keys)
    )
    pull_summaries: Final = tuple(
        _pull_summary(pull, address, method, observed) for pull, address, method in matched_pulls
    )
    eligible_emails: Final = frozenset(person["email"] for person in people if person["eligible"])
    dates: Final = tuple(
        sorted(
            frozenset(row["date"] for row in report["spend"])
            | frozenset(pull["merged_at"][:10] for pull in report["pulls"])
        )
    )
    trend: Final[tuple[ROITrendDay, ...]] = tuple(
        ROITrendDay(
            date=day,
            spend=sum(
                row["spend"]
                for row in report["spend"]
                if row["date"] == day and normalize_email(row["email"]) in eligible_emails
            ),
            hours=sum(
                pull["estimate"]["hours"] or 0.0
                for pull in pull_summaries
                if pull["merged_at"][:10] == day
                and pull["email"] in eligible_emails
                and pull["estimate"]["status"] == "estimated"
            ),
            prs=sum(
                pull["email"] in eligible_emails
                and pull["estimate"]["status"] == "estimated"
                for pull in pull_summaries
                if pull["merged_at"][:10] == day
            ),
        )
        for day in dates
    )
    cohort: Final = tuple(person for person in people if person["eligible"])
    matched_spend: Final = sum(person["spend"] or 0.0 for person in cohort)
    output_hours: Final = sum(person["hours"] for person in cohort)
    total_spend: Final = sum(row["spend"] for row in report["spend"])
    total_output_hours: Final = sum(person["hours"] for person in people)
    metrics: Final = ROISummaryMetrics(
        matched_spend=matched_spend,
        output_hours=output_hours,
        total_spend=total_spend,
        total_output_hours=total_output_hours,
        excluded_spend=max(0.0, total_spend - matched_spend),
        cost_per_hour=matched_spend / output_hours if output_hours else None,
        hours_per_dollar=output_hours / matched_spend if matched_spend else None,
        merged_prs=len(pull_summaries),
        estimated_prs=sum(person["estimated_prs"] for person in people),
        matched_prs=sum(pull["matched"] for pull in pull_summaries),
        cohort_people=len(cohort),
        people_with_prs=sum(person["prs"] > 0 for person in people),
        pending_prs=sum(person["pending_prs"] for person in people),
    )
    summary_people: Final = tuple(
        sorted(people, key=lambda person: (-person["hours"], person["id"]))
    )
    summary_pulls: Final = tuple(
        sorted(pull_summaries, key=lambda pull: pull["merged_at"], reverse=True)
    )
    return ROISummary(
        id=report.get("id"),
        mode=report["mode"],
        start=report["start"],
        end=report["end"],
        synced_at=report["synced_at"],
        repos=report["repos"],
        estimator_model=report["estimator_model"],
        estimator_prompt=report.get("estimator_prompt", ""),
        warnings=report.get("warnings", ()),
        effort_basis=report.get("effort_basis"),
        metrics=metrics,
        people=summary_people,
        pulls=summary_pulls,
        trend=trend,
    )
