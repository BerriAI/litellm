import hashlib
import json
from typing import Final

from litellm.proxy.roi_calculator.github import GitHubPullListItem
from litellm.types.roi_calculator import ROISettings


def cache_key(
    settings: ROISettings,
    context: str,
    repo: str,
    pull: GitHubPullListItem,
) -> str | None:
    head: Final = pull.head.sha if pull.head is not None else ""
    login: Final = pull.user.login if pull.user is not None else ""
    if not head or "body" not in pull.model_fields_set or not login:
        return None
    value: Final = json.dumps(
        (
            "pull-v2-branches",
            settings.source_provider,
            settings.source_api_url.rstrip("/"),
            context,
            repo.casefold() if settings.source_provider == "github" else repo,
            pull.number,
            head,
            pull.head.ref if pull.head is not None else "",
            pull.head.repo.full_name if pull.head is not None and pull.head.repo is not None else "",
            pull.title,
            pull.body or "",
            login.casefold(),
        ),
        ensure_ascii=False,
    )
    return hashlib.sha256(value.encode()).hexdigest()


def settings_fingerprint(settings: ROISettings) -> str:
    value: Final = json.dumps(
        (
            settings.source_provider,
            settings.source_api_url.rstrip("/"),
            settings.repos,
            settings.estimator_model,
            settings.estimator_prompt,
            settings.backfill_days,
        ),
        ensure_ascii=False,
    )
    return hashlib.sha256(value.encode()).hexdigest()
