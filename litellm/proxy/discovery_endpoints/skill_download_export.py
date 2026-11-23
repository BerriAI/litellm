from collections.abc import AsyncIterator
from datetime import timezone
from typing import Annotated, Final
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from litellm.models.plugin_upload import PluginDownloadEvent
from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.csv_export import csv_row
from litellm.proxy.common_utils.resource_ownership import is_proxy_admin
from litellm.proxy.deps.repositories import PluginUploadRepositoryDep

router: Final = APIRouter()
_COLUMNS: Final = ("download_id", "downloaded_at", "user_email", "user_id", "team", "key", "version")


def _event_row(event: PluginDownloadEvent) -> bytes:
    return csv_row(
        (
            event.id,
            event.downloaded_at.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            event.user_email,
            event.user_id,
            event.team_alias or event.team_id,
            event.key_alias,
            event.version,
        )
    )


async def _stream_downloads(
    first: PluginDownloadEvent | None,
    events: AsyncIterator[PluginDownloadEvent],
) -> AsyncIterator[bytes]:
    yield csv_row(_COLUMNS)
    if first is not None:
        yield _event_row(first)
    async for event in events:
        yield _event_row(event)


@router.get(
    "/{name}/downloads/export",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Authenticated download history as CSV",
            "content": {
                "text/csv": {"schema": {"type": "string"}},
            },
        }
    },
)
async def export_skill_downloads(
    name: str,
    uploads: PluginUploadRepositoryDep,
    caller: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> StreamingResponse:
    """Export all recorded authenticated downloads. Anonymous downloads remain totals-only."""
    if not is_proxy_admin(caller):
        raise HTTPException(403, "Only proxy admins may export skill download history")
    if uploads is None:
        raise HTTPException(500, CommonProxyErrors.db_not_connected_error.value)
    plugin: Final = await uploads.get_plugin(name)
    if plugin is None:
        raise HTTPException(404, "Skill not found")
    events: Final = uploads.export_downloads(plugin.id)
    first: Final = await anext(events, None)
    return StreamingResponse(
        _stream_downloads(first, events),
        media_type="text/csv",
        headers={
            "Cache-Control": "no-store",
            "Content-Disposition": f"attachment; filename*=UTF-8''{quote(plugin.name, safe='')}.csv",
        },
    )
