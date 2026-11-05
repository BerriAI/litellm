"""Metadata keys and headers shared by the ORACLE router, its post-call hook and the proxy endpoints."""

from typing import Final

ORACLE_ROUTER_MODEL_PREFIX: Final[str] = "auto_router/oracle_router"

CHOSEN_MODEL_METADATA_KEY: Final[str] = "oracle_router_chosen_model"
PROGRAM_ID_METADATA_KEY: Final[str] = "oracle_router_program_id"
RESPONSE_HEADER: Final[str] = "x-litellm-oracle-router-model"
PROGRAM_ID_HEADER: Final[str] = "x-litellm-program-id"

PROGRAM_DONE_KEY: Final[str] = "program_done"
PROGRAM_SCORE_KEY: Final[str] = "program_score"

SESSION_ID_FALLBACK_KEYS: Final[tuple[str, ...]] = ("litellm_session_id", "session_id")

FEEDBACK_HISTORY_SIZE: Final[int] = 200
PROGRAM_SWEEP_THRESHOLD: Final[int] = 1024
