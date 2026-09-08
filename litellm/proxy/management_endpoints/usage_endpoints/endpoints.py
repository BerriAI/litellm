"""
USAGE AI CHAT ENDPOINTS

/usage/ai/chat - Stream AI chat responses about usage data
"""

from typing import Final, Literal

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.sse_keepalive import (
    SSE_COMMENT_PING,
    wrap_sse_stream_with_keepalive_pings,
)

router: Final = APIRouter()


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class UsageAIChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., description="Chat messages (user/assistant history)")
    model: str | None = Field(default=None, description="Model to use for AI chat")


@router.post(
    "/usage/ai/chat",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
)
async def usage_ai_chat(
    data: UsageAIChatRequest,
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    AI chat about usage data. Streams SSE events with the AI response.
    The AI agent has access to tools that query aggregated daily activity data.
    """
    from litellm.proxy.management_endpoints.common_utils import (
        _user_has_admin_view,
        require_caller_user_id_for_non_admin,
    )
    from litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat import (
        stream_usage_ai_chat,
    )

    is_admin: Final = _user_has_admin_view(user_api_key_dict)
    if is_admin:
        user_id = user_api_key_dict.user_id
    else:
        user_id = require_caller_user_id_for_non_admin(user_api_key_dict)
    messages: Final = [{"role": m.role, "content": m.content} for m in data.messages]

    return StreamingResponse(
        wrap_sse_stream_with_keepalive_pings(
            stream_usage_ai_chat(
                messages=messages,
                model=data.model,
                user_id=user_id,
                is_admin=is_admin,
            ),
            ping_interval_seconds=litellm.sse_keepalive_ping_interval_seconds,
            ping_chunk=SSE_COMMENT_PING,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
