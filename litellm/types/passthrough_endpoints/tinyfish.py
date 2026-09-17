from typing import Final

from typing_extensions import ReadOnly, TypedDict

TINYFISH_AGENT_DEFAULT_API_BASE: Final = "https://agent.tinyfish.ai"
TINYFISH_AGENT_DOCS_URL: Final = "https://docs.tinyfish.ai/agent-api"
# TinyFish's published Agent API rate (USD per run step); override with env TINYFISH_COST_PER_STEP
TINYFISH_DEFAULT_COST_PER_STEP: Final = 0.016
TINYFISH_MODEL_NAME: Final = "tinyfish/automation-run"
TINYFISH_POLLING_INTERVAL_SECONDS: Final = 5.0
# the Agent API caps runs at 1200s but queue wait extends wall time, so billing polls with generous headroom
TINYFISH_MAX_POLLING_SECONDS: Final = 3600.0
# at the 5s interval this tolerates a ~60s upstream outage before abandoning the charge
TINYFISH_MAX_CONSECUTIVE_POLL_FAILURES: Final = 12

TINYFISH_TERMINAL_RUN_STATUSES: Final = frozenset({"COMPLETED", "FAILED", "CANCELLED"})

# these fields use the shared account's saved logins/vault, so they 403 unless TINYFISH_ALLOW_AUTHENTICATED_RUNS=true
TINYFISH_AUTHENTICATED_RUN_FIELDS: Final = frozenset({"use_profile", "profile_id", "use_vault", "credential_item_ids"})

_RUN_SUBMIT_PATHS: Final = frozenset(
    {("v1", "automation", "run"), ("v1", "automation", "run-async"), ("v1", "automation", "run-sse")}
)


class TinyfishRun(TypedDict, total=False):
    """Run objects are null-heavy until terminal, so every field must tolerate None."""

    run_id: ReadOnly[str | None]
    status: ReadOnly[str | None]
    num_of_steps: ReadOnly[int | None]
    result: ReadOnly[object]
    # left untyped on purpose: a strict error shape would fail whole-run validation on upstream drift and drop the charge
    error: ReadOnly[object]
    type: ReadOnly[str | None]


def is_allowed_tinyfish_endpoint(method: str, path: str) -> bool:
    """The host also serves vault/wallet/profile management under the same key, so only run endpoints forward."""
    segments: Final = tuple(part for part in path.split("/") if part)
    if any(segment in (".", "..") for segment in segments):
        return False
    if method == "POST" and segments in _RUN_SUBMIT_PATHS:
        return True
    # no GET /v1/runs listing: run ids are unguessable, so blocking the list keeps teams out of each other's runs
    if method == "GET" and len(segments) == 3 and segments[:2] == ("v1", "runs"):
        return True
    return method == "POST" and len(segments) == 4 and segments[:2] == ("v1", "runs") and segments[3] == "cancel"
