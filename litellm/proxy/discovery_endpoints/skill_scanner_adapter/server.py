# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "fastapi==0.162.0", "uvicorn==0.54.0", "python-multipart==0.0.33",
#   "skillspector @ git+https://github.com/NVIDIA/skillspector.git@62c2fe8622f580ac292d4078c3b71f790194dbbd",
# ]
# ///
import asyncio
import hmac
import json
import os
import signal
import sys
import tempfile
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Final
from urllib.parse import quote, urlsplit

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import AwareDatetime, TypeAdapter, ValidationError

MAX_UPLOAD: Final = 64 * 1024 * 1024
MAX_REPORT: Final = 1024 * 1024
SCAN_TIMEOUT: Final = 570
SOURCE: Final = TypeAdapter(dict[str, str])
SLOTS: Final = asyncio.Semaphore(2)
app: Final = FastAPI(title="SkillSpector adapter", docs_url=None, redoc_url=None, openapi_url=None)


def authenticate(credentials: Annotated[HTTPAuthorizationCredentials, Depends(HTTPBearer())]) -> None:
    token: Final = os.environ.get("SKILL_SCANNER_API_KEY", "")
    if not token or not hmac.compare_digest(credentials.credentials, token):
        raise HTTPException(401, "Invalid scanner credentials")


def source_target(source: dict[str, str]) -> str:
    target: Final = (
        "https://github.com/" + source.get("repo", "") if source.get("source") == "github" else source.get("url", "")
    )
    parsed: Final = urlsplit(target)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
        raise HTTPException(400, "Remote sources require an HTTPS URL without credentials or a fragment")
    ref: Final = source.get("ref")
    subdir: Final = source.get("path")
    if ref or subdir:
        if parsed.hostname != "github.com" or len(parsed.path.strip("/").split("/")) != 2 or not ref:
            raise HTTPException(400, "Upload a ZIP to scan this source's exact ref and subdirectory")
        suffix: Final = "/" + quote(subdir, safe="/") if subdir else ""
        return target.removesuffix(".git").rstrip("/") + "/tree/" + quote(ref, safe="/") + suffix
    return target


async def worker(target: str, output: Path, use_llm: bool) -> None:
    from skillspector.mcp_server import run_scan

    result: Final = await run_scan(target, use_llm=use_llm, allow_local_targets=not target.startswith("https://"))
    coverage: Final = result.get("analysis_completeness", {})
    findings: Final = result["findings"]
    complete: Final = (
        result.get("execution_successful") is True
        and coverage.get("is_complete") is True
        and (not use_llm or (result.get("llm_available") is True and result.get("llm_used") is True))
        and len(findings) <= 500
    )
    limitations: Final = tuple(str(value)[:2000] for value in coverage.get("limitations", ()))
    report: Final = {
        "scanner_version": result["version"],
        "complete": complete,
        "scan_mode": result["scan_mode"],
        "risk_score": result["risk_score"],
        "findings": [
            {
                "rule_id": finding["id"],
                "severity": finding["severity"],
                "message": finding.get("pattern") or finding.get("finding") or "Security finding",
                "file": finding["location"]["file"],
                "start_line": finding["location"].get("start_line"),
                "explanation": finding.get("explanation"),
                "remediation": finding.get("remediation"),
            }
            for finding in findings[:500]
        ],
        "limitations": (
            *limitations,
            *(("Only the first 500 findings are displayed.",) if len(findings) > 500 else ()),
        ),
    }
    encoded: Final = json.dumps(report).encode()
    if len(encoded) > MAX_REPORT:
        raise ValueError("Scanner report exceeded the response limit")
    output.write_bytes(encoded)


async def run_worker(target: str, output: Path, use_llm: bool, expires_at: datetime | None = None) -> dict[str, object]:
    timeout: Final = (
        SCAN_TIMEOUT
        if expires_at is None
        else min(SCAN_TIMEOUT, (expires_at - datetime.now(timezone.utc)).total_seconds())
    )
    if timeout <= 0:
        raise HTTPException(504, "SkillSpector scan deadline expired")
    process: Final = await asyncio.create_subprocess_exec(
        sys.executable,
        str(Path(__file__).resolve()),
        "--scan",
        target,
        str(output),
        str(use_llm).lower(),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        start_new_session=True,
        env={
            **os.environ,
            "HOME": str(output.parent),
            "TMPDIR": str(output.parent),
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_ALLOW_PROTOCOL": "https",
        },
    )
    try:
        code: Final = await asyncio.wait_for(process.wait(), timeout)
        if code != 0 or not output.is_file() or output.stat().st_size > MAX_REPORT:
            raise HTTPException(502, "SkillSpector could not produce a report for this source")
        return TypeAdapter(dict[str, object]).validate_json(output.read_bytes())
    except TimeoutError:
        raise HTTPException(504, "SkillSpector scan timed out") from None
    finally:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        await process.wait()


@app.post("/scan", dependencies=[Depends(authenticate)])
async def scan(
    source: Annotated[str, Form(max_length=16384)],
    use_llm: Annotated[bool, Form()] = False,
    expires_at: Annotated[AwareDatetime | None, Form()] = None,
    file: Annotated[UploadFile | None, File()] = None,
) -> dict[str, object]:
    if SLOTS.locked():
        raise HTTPException(429, "Scanner is busy; try again shortly")
    try:
        parsed: Final = SOURCE.validate_json(source)
    except ValidationError:
        raise HTTPException(400, "Invalid source metadata") from None
    async with SLOTS:
        with tempfile.TemporaryDirectory(prefix="skill-scan-") as directory:
            root: Final = Path(directory)
            if file is not None:
                content: Final = await file.read(MAX_UPLOAD + 1)
                if len(content) > MAX_UPLOAD:
                    raise HTTPException(413, "Upload exceeds 64 MiB")
                archive: Final = root / "skill.zip"
                archive.write_bytes(content)
                return await run_worker(str(archive), root / "report.json", use_llm, expires_at)
            return await run_worker(source_target(parsed), root / "report.json", use_llm, expires_at)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--scan":
        asyncio.run(worker(sys.argv[2], Path(sys.argv[3]), sys.argv[4] == "true"))
    else:
        import uvicorn

        if not os.environ.get("SKILL_SCANNER_API_KEY"):
            raise SystemExit("Set SKILL_SCANNER_API_KEY before starting the scanner")
        uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8099")))
