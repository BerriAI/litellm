from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

ScheduleKind = Literal["interval", "cron", "once"]
TaskStatus = Literal["pending", "fired", "expired", "cancelled"]


class CreateScheduledTaskRequest(BaseModel):
    title: str
    action: str
    action_args: Optional[Dict[str, Any]] = None
    check_prompt: Optional[str] = None
    format_prompt: Optional[str] = None

    schedule_kind: ScheduleKind
    schedule_spec: str
    schedule_tz: Optional[str] = None

    expires_at: datetime
    fire_once: bool = True


class UpdateScheduledTaskRequest(BaseModel):
    title: Optional[str] = None
    check_prompt: Optional[str] = None
    schedule_kind: Optional[ScheduleKind] = None
    schedule_spec: Optional[str] = None
    schedule_tz: Optional[str] = None
    next_run_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    fire_once: Optional[bool] = None
    action: Optional[str] = None
    action_args: Optional[Dict[str, Any]] = None
    format_prompt: Optional[str] = None


class ScheduledTaskResponse(BaseModel):
    task_id: str
    owner_token: str
    user_id: Optional[str]
    team_id: Optional[str]
    agent_id: Optional[str]

    title: str
    action: str
    action_args: Optional[Dict[str, Any]]
    check_prompt: Optional[str]
    format_prompt: Optional[str]

    schedule_kind: str
    schedule_spec: str
    schedule_tz: Optional[str]

    next_run_at: datetime
    expires_at: datetime
    fire_once: bool

    status: str
    last_fired_at: Optional[datetime]

    created_at: datetime
    updated_at: datetime


class DueTaskResponse(BaseModel):
    """Trimmed shape returned by /due — only fields the agent needs to dispatch."""

    task_id: str
    title: str
    action: str
    action_args: Optional[Dict[str, Any]]
    check_prompt: Optional[str]
    format_prompt: Optional[str]
    scheduled_for: datetime


class ListScheduledTasksResponse(BaseModel):
    tasks: List[ScheduledTaskResponse]


class DueTasksResponse(BaseModel):
    tasks: List[DueTaskResponse] = Field(default_factory=list)
