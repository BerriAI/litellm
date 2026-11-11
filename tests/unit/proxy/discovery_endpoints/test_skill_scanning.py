import hashlib
import io
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from litellm.models.skill_submission import SkillScanInput
from litellm.proxy.discovery_endpoints.skill_scanning import scan_report
from litellm.types.proxy.skill_scanning import SkillScan, SkillScannerConfig, SkillScanReport

NOW: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)
CONFIG: Final = SkillScannerConfig(endpoint="https://scanner.example/scan", api_key_env="SCANNER_TEST_KEY")
REPORT: Final = SkillScanReport(
    scanner_version="test-version", complete=True, scan_mode="static-only", risk_score=0, findings=(),
)
TARGET: Final = SkillScanInput(
    SkillScan(id="scan-1", status="running", requested_by="admin", started_at=NOW, expires_at=NOW + timedelta(seconds=10)),
    {"source": "archive", "url": "https://gateway.example/private.zip"},
    b"exact-private-archive",
)


@pytest.mark.asyncio
async def test_scan_posts_stored_bytes_with_only_scanner_credentials() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url == CONFIG.endpoint
        assert request.headers["authorization"] == "Bearer scanner-only-token"
        assert b"exact-private-archive" in request.content
        assert b'name="use_llm"\r\n\r\nfalse' in request.content
        assert f'name="expires_at"\r\n\r\n{TARGET.scan.expires_at.isoformat()}'.encode() in request.content
        return httpx.Response(200, json=REPORT.model_dump())

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result: Final = await scan_report(client, CONFIG, "scanner-only-token", TARGET)
    assert result == REPORT


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ("redirect", "unavailable", "malformed", "oversized", "network"))
async def test_scanner_failures_never_become_clean_reports(failure: str) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "scanner.example"
        if failure == "redirect":
            return httpx.Response(302, headers={"Location": "https://untrusted.example/scan"})
        if failure == "unavailable":
            return httpx.Response(503, text="provider-internal-secret")
        if failure == "malformed":
            return httpx.Response(200, json={"complete": True})
        if failure == "oversized":
            return httpx.Response(200, content=b" " * (1024 * 1024 + 1))
        raise httpx.ConnectError("provider-internal-secret", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond), follow_redirects=True) as client:
        result: Final = await scan_report(client, CONFIG, "scanner-only-token", TARGET)
    assert isinstance(result, str)
    assert "provider-internal-secret" not in result


@pytest.mark.asyncio
@pytest.mark.parametrize("existing_limitations", (0, 100), ids=("empty", "full"))
async def test_missing_requested_semantic_pass_remains_incomplete(existing_limitations: int) -> None:
    report: Final = REPORT.model_copy(update={"limitations": ("Existing limitation",) * existing_limitations})

    def respond(request: httpx.Request) -> httpx.Response:
        assert b'name="use_llm"\r\n\r\ntrue' in request.content
        return httpx.Response(200, json=report.model_dump())

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result: Final = await scan_report(client, CONFIG.model_copy(update={"use_llm": True}), "token", TARGET)
    assert isinstance(result, SkillScanReport)
    assert result == report.model_copy(update={
        "complete": False, "limitations": (*report.limitations[:99], "Requested semantic analysis did not run."),
    })
    assert SkillScanReport.model_validate(result.model_dump()) == result


def test_interrupted_scan_becomes_retryable_without_changing_completed_evidence() -> None:
    assert TARGET.scan.at(NOW) == TARGET.scan
    assert TARGET.scan.at(NOW + timedelta(seconds=10)).status == "failed"
    complete: Final = TARGET.scan.model_copy(update={"status": "completed", "report": REPORT})
    assert complete.at(NOW + timedelta(days=1)) == complete


@pytest.mark.parametrize("endpoint", (
    "http://scanner.example/scan", "https://user:secret@scanner.example/scan", "https://scanner.example/scan?token=secret",
))
def test_scanner_config_rejects_unsafe_credential_destinations(endpoint: str) -> None:
    with pytest.raises(ValidationError):
        SkillScannerConfig(endpoint=endpoint, api_key_env="SCANNER_TEST_KEY")


def test_adapter_authentication_and_exact_remote_source(monkeypatch: pytest.MonkeyPatch) -> None:
    from cookbook.skill_scanner import server

    monkeypatch.setenv("SKILL_SCANNER_API_KEY", "adapter-secret")
    clock: Final = MagicMock()
    clock.now.return_value = NOW + timedelta(seconds=1)
    monkeypatch.setattr(server, "datetime", clock)
    launch: Final = AsyncMock(side_effect=AssertionError("Invalid deadline launched a worker"))
    monkeypatch.setattr(server.asyncio, "create_subprocess_exec", launch)
    with TestClient(server.app) as client:
        assert client.post("/scan", data={"source": "{}"}).status_code == 401
        assert client.post("/scan", headers={"Authorization": "Bearer wrong"}, data={"source": "{}"}).status_code == 401
        response = client.post("/scan", headers={"Authorization": "Bearer adapter-secret"}, data={
            "source": '{"source":"git","url":"https://example.org/repo.git","ref":"release"}',
        })
        assert response.status_code == 400
        for deadline, status in ((NOW.isoformat(), 504), (NOW.replace(tzinfo=None).isoformat(), 422)):
            rejected: Final = client.post("/scan", headers={"Authorization": "Bearer adapter-secret"}, data={
                "source": '{"source":"archive","url":"https://example.org/skill.zip"}', "expires_at": deadline,
            })
            assert rejected.status_code == status
    launch.assert_not_called()
    assert server.source_target({"source": "archive", "url": "https://example.org/skill.zip"}) == "https://example.org/skill.zip"
    assert server.source_target({"source": "github", "repo": "example/skills", "ref": "v2", "path": "skills/review"}) == "https://github.com/example/skills/tree/v2/skills/review"


def test_adapter_accepts_expanded_stored_skill_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    from cookbook.skill_scanner import server
    from litellm.proxy.anthropic_endpoints.claude_code_endpoints.skill_upload_archive import (
        build_uploaded_skill_archive,
    )

    uploaded: Final = io.BytesIO()
    with zipfile.ZipFile(uploaded, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("SKILL.md", "---\nname: large-skill\ndescription: Synthetic archive\n---\n")
        for name in ("first.bin", "second.bin"):
            archive.writestr(name, b"x" * (6 * 1024 * 1024))
    stored: Final = build_uploaded_skill_archive(uploaded.getvalue(), "large-skill")
    assert stored is not None
    stored_size: Final = len(stored)
    assert stored_size > 10 * 1024 * 1024
    expected: Final = (stored_size, hashlib.sha256(stored).hexdigest())

    async def run_worker(
        target: str, output: Path, use_llm: bool, expires_at: datetime | None = None,
    ) -> dict[str, object]:
        received: Final = Path(target).read_bytes()
        actual: Final = (len(received), hashlib.sha256(received).hexdigest())
        assert actual == expected
        return {"complete": True}

    monkeypatch.setattr(server, "run_worker", run_worker)
    monkeypatch.setenv("SKILL_SCANNER_API_KEY", "adapter-secret")
    with TestClient(server.app) as client:
        response: Final = client.post("/scan", headers={"Authorization": "Bearer adapter-secret"},
                                      data={"source": "{}"}, files={"file": ("skill.zip", stored, "application/zip")})
    assert response.status_code == 200 and response.json() == {"complete": True}
