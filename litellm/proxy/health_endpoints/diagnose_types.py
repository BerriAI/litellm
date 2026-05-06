from typing import Optional

from pydantic import BaseModel, Field


class DiagnoseRequest(BaseModel):
    model: Optional[str] = Field(
        default=None,
        description=(
            "Optional proxy model name to use for generating the diagnostic report. "
            "When omitted, LiteLLM uses the first configured proxy deployment."
        ),
    )
    issue_description: Optional[str] = Field(
        default=None,
        description="Short description of the issue the admin is trying to reproduce.",
    )
    reproduction_steps: Optional[str] = Field(
        default=None,
        description="Known reproduction steps or the behavior the admin has already tried.",
    )
    diagnostic_answers: Optional[list[str]] = Field(
        default=None,
        description=(
            "Answers collected so far. /diagnose returns one next_question at a time "
            "plus next_request_body/next_curl showing how to send the next answer. "
            "When at least three answers are provided, LiteLLM generates the final "
            "support report."
        ),
    )
    diagnostic_session_id: Optional[str] = Field(
        default=None,
        description=(
            "Opaque question-state token returned by /diagnose. Pass it back from "
            "next_request_body/next_curl so LiteLLM can ask one question per request "
            "without exposing future questions early."
        ),
    )
