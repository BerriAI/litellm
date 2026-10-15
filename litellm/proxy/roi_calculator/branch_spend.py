import json
from collections import Counter
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta, timezone
from typing import Final, Protocol

from pydantic import TypeAdapter

from litellm.types.roi_calculator import ROIBranchAttribution, ROIBranchSpend, ROIPullRecord


class BranchSpendDatabase(Protocol):
    async def query_raw(self, query: str, *args: object) -> object: ...


async def read_branch_spend(
    database: BranchSpendDatabase, start: date, end: date, repos: tuple[str, ...], *, casefold_repo: bool = False
) -> tuple[ROIBranchSpend, ...]:
    if not repos:
        return ()
    query: Final = """
        WITH tagged AS (
            SELECT logs.spend, tags.repos[1] AS repo, tags.branches[1] AS branch
            FROM "LiteLLM_SpendLogs" AS logs
            CROSS JOIN LATERAL (
                SELECT array_agg(DISTINCT substring(tag FROM 6))
                           FILTER (WHERE starts_with(tag, 'repo:')) AS repos,
                       array_agg(DISTINCT substring(tag FROM 8))
                           FILTER (WHERE starts_with(tag, 'branch:')) AS branches
                FROM jsonb_array_elements_text(
                    CASE WHEN jsonb_typeof(logs.request_tags) = 'array'
                         THEN logs.request_tags ELSE '[]'::jsonb END
                ) AS tag
            ) AS tags
            WHERE logs."startTime" >= $1::text::timestamp AND logs."startTime" < $2::text::timestamp
              AND cardinality(tags.repos) = 1 AND cardinality(tags.branches) = 1
              AND CASE logs.metadata -> 'litellm_roi_estimator'
                  WHEN 'true'::jsonb THEN false
                  WHEN 'false'::jsonb THEN true
                  ELSE NOT coalesce(logs.request_tags ? 'litellm-roi-estimator', false)
              END
        )
        SELECT CASE WHEN $4 THEN lower(repo) ELSE repo END AS repo,
               branch, sum(spend)::double precision AS spend, count(*)::integer AS requests
        FROM tagged
        WHERE branch <> '' AND (CASE WHEN $4 THEN lower(repo) ELSE repo END)
              IN (SELECT jsonb_array_elements_text($3::jsonb))
        GROUP BY 1, 2
        ORDER BY 1, 2
    """
    result: Final = await database.query_raw(
        query,
        datetime.combine(start, time.min, timezone.utc).isoformat(),
        datetime.combine(end + timedelta(days=1), time.min, timezone.utc).isoformat(),
        json.dumps(repos),
        casefold_repo,
    )
    return TypeAdapter(tuple[ROIBranchSpend, ...]).validate_python(result)


def attribute_branches(
    pulls: tuple[ROIPullRecord, ...], spend: tuple[ROIBranchSpend, ...] | None
) -> Mapping[tuple[str, int], ROIBranchAttribution]:
    return attribute_branch_keys(
        tuple(
            (pull["repo"], pull["number"], pull.get("source_repo", ""), pull.get("source_branch", "")) for pull in pulls
        ),
        spend,
    )


def attribute_branch_keys(
    pulls: tuple[tuple[str, int, str, str], ...], spend: tuple[ROIBranchSpend, ...] | None
) -> Mapping[tuple[str, int], ROIBranchAttribution]:
    counts: Final = Counter((pull[2], pull[3]) for pull in pulls)
    costs: Final = {(row.repo, row.branch): row for row in spend or ()}

    def attribute(pull: tuple[str, int, str, str]) -> ROIBranchAttribution:
        repo: Final = pull[2]
        branch: Final = pull[3]
        cost: Final = costs.get((repo, branch))
        if spend is None:
            return ROIBranchAttribution(repo=repo, branch=branch, status="unavailable")
        if not repo or not branch or cost is None:
            return ROIBranchAttribution(repo=repo, branch=branch)
        if counts[(repo, branch)] != 1:
            return ROIBranchAttribution(repo=repo, branch=branch, status="ambiguous")
        return ROIBranchAttribution(
            repo=repo, branch=branch, spend=cost.spend, requests=cost.requests, status="matched"
        )

    return {(pull[0], pull[1]): attribute(pull) for pull in pulls}
