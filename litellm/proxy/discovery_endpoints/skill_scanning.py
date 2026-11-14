import asyncio
import io
import json
from datetime import datetime, timedelta, timezone
from typing import Annotated, Final
from uuid import uuid4

import httpx
from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException
from pydantic import ValidationError

from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # shared HTTP factory has untyped optional params
)
from litellm.models.skill_submission import SkillScanInput, SkillSubmissionStore, SubmissionConflict, SubmissionMissing
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.resource_ownership import is_proxy_admin
from litellm.proxy.deps.repositories import SkillSubmissionRepositoryDep
from litellm.secret_managers.main import get_secret_str
from litellm.types.proxy.skill_scanning import (
    SkillScan,
    SkillScannerConfig,
    SkillScanReport,
    SkillScanRequest,
    SkillScanResponse,
)

router: Final = APIRouter()
_MAX_REPORT_BYTES: Final = 1024 * 1024


def scanner_config() -> SkillScannerConfig | str | None:
    from litellm.proxy.proxy_server import general_settings_view

    value: Final = general_settings_view().get("skill_scanner")
    if value is None:
        return None
    try:
        return SkillScannerConfig.model_validate(value)
    except ValidationError:
        return "The skill scanner configuration is invalid"


async def scan_report(
    client: httpx.AsyncClient, config: SkillScannerConfig, api_key: str, target: SkillScanInput
) -> SkillScanReport | str:
    try:
        async with client.stream(
            "POST",
            config.endpoint,
            headers={"Authorization": f"Bearer {api_key}", "Accept-Encoding": "identity"},
            data={
                "source": json.dumps(dict(target.source)),
                "use_llm": str(config.use_llm).lower(),
                "expires_at": target.scan.expires_at.isoformat(),
            },
            files=None if target.content is None else {"file": ("skill.zip", target.content, "application/zip")},
            timeout=httpx.Timeout(config.timeout_seconds, connect=10),
            follow_redirects=False,
        ) as response:
            if response.status_code != 200:
                return f"Scanner returned HTTP {response.status_code}. Check the scanner service and authentication."
            output: Final = io.BytesIO()
            async for chunk in response.aiter_bytes(chunk_size=65536):
                if output.tell() + len(chunk) > _MAX_REPORT_BYTES:
                    return "Scanner report exceeded the 1 MiB limit."
                output.write(chunk)
            report: Final = SkillScanReport.model_validate_json(output.getvalue())
            if config.use_llm and report.scan_mode != "static+llm":
                return report.model_copy(
                    update={
                        "complete": False,
                        "limitations": (*report.limitations[:99], "Requested semantic analysis did not run."),
                    }
                )
            return report
    except (httpx.HTTPError, ValidationError):
        return "The scanner was unreachable or returned an invalid report."


async def complete_scan(
    submissions: SkillSubmissionStore,
    submission_id: str,
    target: SkillScanInput,
    config: SkillScannerConfig,
    api_key: str,
) -> None:
    try:
        client: Final = get_async_httpx_client(
            "skill_scanner", params={"ssl_verify": True, "follow_redirects": False}
        ).client
        report: Final = await asyncio.wait_for(
            scan_report(client, config, api_key, target),
            timeout=(target.scan.expires_at - datetime.now(timezone.utc)).total_seconds(),
        )
        completed: Final = target.scan.model_copy(
            update={
                "status": "completed" if isinstance(report, SkillScanReport) else "failed",
                "report": report if isinstance(report, SkillScanReport) else None,
                "error": report if isinstance(report, str) else None,
                "completed_at": datetime.now(timezone.utc),
            }
        )
        await submissions.finish_scan(submission_id, completed)
    except (TimeoutError, asyncio.CancelledError) as error:
        await submissions.finish_scan(
            submission_id,
            target.scan.model_copy(
                update={
                    "status": "failed",
                    "error": "Scan timed out or was interrupted. Try scanning again.",
                    "completed_at": datetime.now(timezone.utc),
                }
            ),
        )
        if isinstance(error, asyncio.CancelledError):
            raise


@router.get(
    "/skill-hub/submissions/{submission_id}/scan",
    tags=["Skills Hub"],
    response_model=SkillScanResponse,
)
async def get_skill_scan(
    submission_id: str,
    submissions: SkillSubmissionRepositoryDep,
    caller: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> SkillScanResponse:
    """Read the latest manual security scan. Only proxy admins may read reports."""
    if not is_proxy_admin(caller):
        raise HTTPException(403, "Proxy admin access is required to review security scans")
    if submissions is None:
        raise HTTPException(503, "Database is not connected")
    scan: Final = await submissions.get_scan(submission_id)
    if isinstance(scan, SubmissionMissing):
        raise HTTPException(404, "Skill submission not found")
    return SkillScanResponse(
        configured=isinstance(scanner_config(), SkillScannerConfig),
        scan=None if scan is None else scan.at(datetime.now(timezone.utc)),
    )


@router.post(
    "/skill-hub/submissions/{submission_id}/scan",
    tags=["Skills Hub"],
    response_model=SkillScanResponse,
)
async def start_skill_scan(
    submission_id: str,
    background_tasks: BackgroundTasks,
    submissions: SkillSubmissionRepositoryDep,
    caller: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    _request: Annotated[SkillScanRequest | None, Body()] = None,
) -> SkillScanResponse:
    """Start an optional scan of a pending submission. Scans never change approval decisions."""
    if not is_proxy_admin(caller):
        raise HTTPException(403, "Proxy admin access is required to scan skills")
    if submissions is None:
        raise HTTPException(503, "Database is not connected")
    config: Final = scanner_config()
    if config is None:
        raise HTTPException(503, "Security scanning is not configured")
    if isinstance(config, str):
        raise HTTPException(503, config)
    api_key: Final = get_secret_str(config.api_key_env)
    if not api_key:
        raise HTTPException(503, "The scanner authentication secret is not configured")
    now: Final = datetime.now(timezone.utc)
    target: Final = await submissions.start_scan(
        submission_id,
        SkillScan(
            id=str(uuid4()),
            status="running",
            requested_by=caller.user_id or "proxy-admin",
            started_at=now,
            expires_at=now + timedelta(seconds=config.timeout_seconds),
        ),
    )
    if isinstance(target, SubmissionMissing):
        raise HTTPException(404, "Skill submission not found")
    if isinstance(target, SubmissionConflict):
        raise HTTPException(409, target.message)
    background_tasks.add_task(complete_scan, submissions, submission_id, target, config, api_key)
    return SkillScanResponse(configured=True, scan=target.scan)
