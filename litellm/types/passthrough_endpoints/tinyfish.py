from typing import Final

from typing_extensions import ReadOnly, TypedDict

TINYFISH_AGENT_DEFAULT_API_BASE: Final = "https://agent.tinyfish.ai"
TINYFISH_AGENT_DOCS_URL: Final = "https://docs.tinyfish.ai/agent-api"
# TinyFish's published Agent API rate (USD per run step); override with env TINYFISH_COST_PER_STEP
TINYFISH_DEFAULT_COST_PER_STEP: Final = 0.016
TINYFISH_MODEL_NAME: Final = "tinyfish/automation-run"
TINYFISH_POLLING_INTERVAL_SECONDS: Final = 5.0
# Matches the Agent API's 1200s max run duration; raise together or long runs go unbilled
TINYFISH_MAX_POLLING_SECONDS: Final = 1200.0
TINYFISH_MAX_CONSECUTIVE_POLL_FAILURES: Final = 3

TINYFISH_TERMINAL_RUN_STATUSES: Final = frozenset({"COMPLETED", "FAILED", "CANCELLED"})

# Fields that run with the TinyFish account's saved logins/vault; all proxy callers share
# one upstream key, so these are rejected unless TINYFISH_ALLOW_AUTHENTICATED_RUNS=true.
TINYFISH_AUTHENTICATED_RUN_FIELDS: Final = frozenset({"use_profile", "profile_id", "use_vault", "credential_item_ids"})

_RUN_SUBMIT_PATHS: Final = frozenset(
    {("v1", "automation", "run"), ("v1", "automation", "run-async"), ("v1", "automation", "run-sse")}
)


class TinyfishRunError(TypedDict, total=False):
    code: ReadOnly[str | None]
    message: ReadOnly[str | None]
    category: ReadOnly[str | None]
    retry_after: ReadOnly[float | None]
    help_url: ReadOnly[str | None]


class TinyfishRun(TypedDict, total=False):
    """Run objects are null-heavy until terminal, so every field must tolerate None."""

    run_id: ReadOnly[str | None]
    status: ReadOnly[str | None]
    num_of_steps: ReadOnly[int | None]
    result: ReadOnly[object]
    error: ReadOnly[TinyfishRunError | None]
    type: ReadOnly[str | None]


def is_allowed_tinyfish_endpoint(method: str, path: str) -> bool:
    """agent.tinyfish.ai also serves vault/wallet/browser-profile management under the
    same key, so only the run endpoints may be forwarded."""
    segments: Final = tuple(part for part in path.split("/") if part)
    if any(segment in (".", "..") for segment in segments):
        return False
    if method == "POST" and segments in _RUN_SUBMIT_PATHS:
        return True
    # no GET /v1/runs listing: run ids are unguessable, so withholding the list keeps teams
    # behind the shared key from discovering (then reading/cancelling) each other's runs
    if method == "GET" and len(segments) == 3 and segments[:2] == ("v1", "runs"):
        return True
    return method == "POST" and len(segments) == 4 and segments[:2] == ("v1", "runs") and segments[3] == "cancel"
