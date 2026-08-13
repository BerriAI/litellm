from datetime import datetime

from pydantic import BaseModel


class CLISessionResponse(BaseModel):
    """An operator-visible `lite login` session.

    ``session_id`` is the sha256 of the session token, so it identifies the session
    without being usable as a credential.
    """

    session_id: str
    user_id: str
    team_id: str | None = None
    created_at: datetime
    expires_at: datetime
    revoked_at: datetime | None = None
    revoked_by: str | None = None


class CLISessionListResponse(BaseModel):
    sessions: tuple[CLISessionResponse, ...]
    total_count: int
