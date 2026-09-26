from typing import Final

from litellm.proxy.db.queries import load

ADVANCE_BASELINE_COMPARISON_REVISION: Final = load(__name__, "advance_baseline_comparison_revision")
APPLY_BASELINE_SESSION_CORRECTIONS: Final = load(__name__, "apply_baseline_session_corrections")
APPLY_BASELINE_USER_SESSION_CORRECTIONS: Final = load(__name__, "apply_baseline_user_session_corrections")
CLAIM_DIRTY_BASELINE_COMPARISONS: Final = load(__name__, "claim_dirty_baseline_comparisons")
CREATE_BASELINE_COMPARISON: Final = load(__name__, "create_baseline_comparison")
DELETE_RETIRED_BASELINE_OBSERVATIONS: Final = load(__name__, "delete_retired_baseline_observations")
FIND_BASELINE_OBSERVATION_CHANGED_BEFORE: Final = load(__name__, "find_baseline_observation_changed_before")
FIND_BASELINE_OBSERVATION_WITHOUT_SPEND_LOG: Final = load(__name__, "find_baseline_observation_without_spend_log")
INSERT_BASELINE_OBSERVATION: Final = load(__name__, "insert_baseline_observation")
LOCK_BASELINE_COMPARISON: Final = load(__name__, "lock_baseline_comparison")
MARK_BASELINE_OBSERVATION_CONFLICTED: Final = load(__name__, "mark_baseline_observation_conflicted")
PUBLISH_BASELINE_COMPARISON_HISTORY: Final = load(__name__, "publish_baseline_comparison_history")
PUBLISH_BASELINE_TO_SPEND_LOGS: Final = load(__name__, "publish_baseline_to_spend_logs")
READ_BASELINE_OBSERVATION: Final = load(__name__, "read_baseline_observation")
READ_BASELINE_OBSERVATION_PAGE: Final = load(__name__, "read_baseline_observation_page")
RETIRE_EXPIRED_BASELINE_COMPARISONS: Final = load(__name__, "retire_expired_baseline_comparisons")
STORE_BASELINE_PUBLICATIONS: Final = load(__name__, "store_baseline_publications")
