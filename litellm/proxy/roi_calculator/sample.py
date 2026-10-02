from datetime import datetime, timedelta
from typing import Final

from litellm.types.roi_calculator import DEFAULT_PROMPT, ROIEstimate, ROIPullRecord, ROIReport, ROISpendRecord


def sample_report(now: datetime) -> ROIReport:
    start: Final = now.date() - timedelta(days=29)
    examples: Final = (
        ("alex", "alex@example.com", "Add usage breakdown by model", 6.5, 18.2),
        ("jordan", "jordan@example.com", "Fix streaming response cancellation", 4.0, 12.8),
        ("casey", "", "Add integration tests for billing", 5.5, 0.0),
    )

    def pull(index: int, login: str, email: str, title: str, hours: float) -> ROIPullRecord:
        estimate: Final[ROIEstimate] = {
            "status": "estimated",
            "hours": hours,
            "reasoning": "Sample estimate of engineering effort without AI assistance. Live estimates use PR descriptions, file change counts, and commit metadata.",
            "model": "your-estimator-model",
            "effort_basis": "without_ai",
            "evidence_source": "pr_metadata",
            "cached": False,
        }
        return ROIPullRecord(
            repo="example/gateway",
            number=142 + index,
            title=title,
            url="",
            login=login,
            emails=(email,) if email else (),
            profile_email=email,
            merged_at=(start + timedelta(days=2 + index * 2)).isoformat() + "T14:20:00Z",
            head_sha=f"sample-{index}",
            additions=47 + index * 23,
            deletions=12 + index * 4,
            changed_files=3,
            commit_count=1,
            incomplete_metadata=False,
            estimate=estimate,
            cache_key=None,
        )

    pulls: Final = tuple(
        pull(index, login, email, title, hours) for index, (login, email, title, hours, _) in enumerate(examples)
    )
    spend: Final = tuple(
        ROISpendRecord(date=pulls[index]["merged_at"][:10], user_id=login, email=email, spend=cost, requests=150)
        for index, (login, email, _, _, cost) in enumerate(examples)
        if email
    )
    return ROIReport(
        mode="demo",
        start=start.isoformat(),
        end=now.date().isoformat(),
        synced_at=now.isoformat(),
        repos=("example/gateway",),
        estimator_model="your-estimator-model",
        estimator_prompt=DEFAULT_PROMPT,
        effort_basis="without_ai",
        spend=spend,
        pulls=pulls,
        settings_fingerprint="sample",
    )
