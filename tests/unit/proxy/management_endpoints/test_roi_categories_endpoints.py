import base64
from typing import Final

import pytest
from fastapi import HTTPException

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.management_endpoints.roi_categories_endpoints import admin, extract_context
from litellm.proxy.roi_categories.models import ContextUpload


@pytest.mark.asyncio
@pytest.mark.parametrize("role", [LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY])
async def test_non_admin_cannot_read_uploaded_context_or_examples(role: LitellmUserRoles) -> None:
    with pytest.raises(HTTPException) as error:
        await admin(UserAPIKeyAuth(user_role=role))
    assert error.value.status_code == 403


@pytest.mark.asyncio
async def test_context_upload_preserves_text_for_discovery() -> None:
    result: Final = await extract_context(
        ContextUpload(name="business.md", content_base64=base64.b64encode(b"Support resolves tickets").decode()),
        UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
    )
    assert result.name == "business.md"
    assert result.text == "Support resolves tickets"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,content,status",
    [("data.exe", b"text", 415), ("context.txt", b"\xff", 422), ("context.txt", b"x" * (1024 * 1024 + 1), 413)],
)
async def test_invalid_context_files_are_rejected(name: str, content: bytes, status: int) -> None:
    with pytest.raises(HTTPException) as error:
        await extract_context(
            ContextUpload(name=name, content_base64=base64.b64encode(content).decode()),
            UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
        )
    assert error.value.status_code == status
