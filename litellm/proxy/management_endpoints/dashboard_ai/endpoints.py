"""
DASHBOARD AI CHAT ENDPOINT

/dashboard/ai/chat - Stream AI chat responses about usage data
"""

from typing import List, Literal

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter()


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class DashboardAIChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="Chat messages (user/assistant history)")
    model: str | None = Field(default=None, description="Model group to use for AI chat")


@router.post(
    "/dashboard/ai/chat",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
    include_in_schema=False,
)
async def dashboard_ai_chat(
    data: DashboardAIChatRequest,
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    AI chat about usage data. Streams SSE events with the AI response.

    The agent queries aggregated daily activity data through a provider scoped
    to the caller: admins get a global view, non-admins are restricted to their
    own ``user_id``.
    """
    from litellm.proxy._types import user_api_key_has_admin_view
    from litellm.proxy.management_endpoints.common_utils import (
        require_caller_user_id_for_non_admin,
    )
    from litellm.proxy.management_endpoints.dashboard_ai.agent import (
        stream_dashboard_ai_chat,
    )
    from litellm.proxy.management_endpoints.dashboard_ai.scoped_data import (
        AdminScope,
        ScopedUsageDataProvider,
        UserScope,
    )
    from litellm.proxy.proxy_server import prisma_client

    if user_api_key_has_admin_view(user_api_key_dict):
        scope = AdminScope(caller_user_id=user_api_key_dict.user_id)
    else:
        scope = UserScope(user_id=require_caller_user_id_for_non_admin(user_api_key_dict))

    provider = ScopedUsageDataProvider(scope=scope, prisma_client=prisma_client)
    messages = [{"role": m.role, "content": m.content} for m in data.messages]

    return StreamingResponse(
        stream_dashboard_ai_chat(provider=provider, messages=messages, model=data.model),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
