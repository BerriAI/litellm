"""Lightweight RPC server that runs mini-agent tools inside host or Docker."""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any, Dict
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import Optional
import subprocess, datetime, os


class ExecReq(BaseModel):
    language: str
    code: str
    timeout_sec: float = 10.0


class ExecResp(BaseModel):
    ok: bool
    rc: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None
    t_ms: float = 0.0


app = FastAPI()


@app.get("/health")
async def health():
    return {"ok": True}

@app.get("/__meta")
async def meta():
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=os.getcwd()
        ).decode().strip()
    except Exception:
        sha = None
    return {"git_sha": sha, "started_at": datetime.datetime.utcnow().isoformat() + "Z"}


async def _run(cmd: list[str], input_bytes: bytes | None, timeout: float) -> tuple[int, str, str, float]:
    t0 = time.perf_counter()
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE if input_bytes else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(input_bytes), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return -1, "", f"timeout after {timeout}s", (time.perf_counter() - t0) * 1000.0
        rc = proc.returncode or 0
        return (
            rc,
            (out or b"").decode("utf-8", errors="replace"),
            (err or b"").decode("utf-8", errors="replace"),
            (time.perf_counter() - t0) * 1000.0,
        )
    except Exception as e:
        return -2, "", str(e), (time.perf_counter() - t0) * 1000.0



async def exec_code(req: ExecReq):
    lang = (req.language or "").strip().lower()
    code = req.code
    timeout = max(1.0, float(req.timeout_sec or 10.0))

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        if lang in ("py", "python"):
            p = td_path / "main.py"
            p.write_text(code)
            rc, out, err, t_ms = await _run(["python3", str(p)], None, timeout)
            return ExecResp(ok=(rc == 0), rc=rc, stdout=out, stderr=err, error=None if rc == 0 else err, t_ms=t_ms)
        if lang in ("js", "javascript"):
            p = td_path / "main.js"
            p.write_text(code)
            rc, out, err, t_ms = await _run(["node", str(p)], None, timeout)
            return ExecResp(ok=(rc == 0), rc=rc, stdout=out, stderr=err, error=None if rc == 0 else err, t_ms=t_ms)
        if lang in ("ts", "typescript"):
            # Ensure Node types are available for standard libs like 'assert'.
            (td_path / "tsconfig.json").write_text(
                '{"compilerOptions":{'
                '"strict":true,'
                '"target":"ES2020",'
                '"module":"commonjs",'
                '"moduleResolution":"node",'
                '"types":["node"],'
                '"lib":["es2020"]'
                '}}'
            )
            p = td_path / "main.ts"
            p.write_text(code)
            # Type-check first (no emit) to catch issues early.
            rc1, out1, err1, _ = await _run(["tsc", "--pretty", "false", "--noEmit", str(p)], None, timeout)
            if rc1 != 0:
                return ExecResp(ok=False, rc=rc1, stdout=out1, stderr=err1, error=err1, t_ms=0.0)
            # Execute quickly using ts-node with transpile-only (we already type-checked above)
            rc, out, err, t_ms = await _run(["ts-node", "--transpile-only", str(p)], None, timeout)
            return ExecResp(ok=(rc == 0), rc=rc, stdout=out, stderr=err, error=None if rc == 0 else err, t_ms=t_ms)
        if lang in ("c",):
            p = td_path / "main.c"
            p.write_text(code)
            rc1, out1, err1, _ = await _run(["gcc", "-O2", "-Wall", "-Werror", str(p), "-o", str(td_path / "a.out")], None, timeout)
            if rc1 != 0:
                return ExecResp(ok=False, rc=rc1, stdout=out1, stderr=err1, error=err1, t_ms=0.0)
            rc, out, err, t_ms = await _run([str(td_path / "a.out")], None, timeout)
            return ExecResp(ok=(rc == 0), rc=rc, stdout=out, stderr=err, error=None if rc == 0 else err, t_ms=t_ms)
        if lang in ("cc", "cpp", "c++"):
            p = td_path / "main.cpp"
            p.write_text(code)
            rc1, out1, err1, _ = await _run(["g++", "-O2", "-std=c++20", "-Wall", "-Werror", str(p), "-o", str(td_path / "a.out")], None, timeout)
            if rc1 != 0:
                return ExecResp(ok=False, rc=rc1, stdout=out1, stderr=err1, error=err1, t_ms=0.0)
            rc, out, err, t_ms = await _run([str(td_path / "a.out")], None, timeout)
            return ExecResp(ok=(rc == 0), rc=rc, stdout=out, stderr=err, error=None if rc == 0 else err, t_ms=t_ms)
        if lang in ("go",):
            p = td_path / "main.go"
            p.write_text(code)
            rc, out, err, t_ms = await _run(["go", "run", str(p)], None, timeout)
            return ExecResp(ok=(rc == 0), rc=rc, stdout=out, stderr=err, error=None if rc == 0 else err, t_ms=t_ms)
        if lang in ("java",):
            p = td_path / "Main.java"
            p.write_text(code)
            rc1, out1, err1, _ = await _run(["javac", str(p)], None, timeout)
            if rc1 != 0:
                return ExecResp(ok=False, rc=rc1, stdout=out1, stderr=err1, error=err1, t_ms=0.0)
            rc, out, err, t_ms = await _run(["java", "-cp", str(td_path), "Main"], None, timeout)
            return ExecResp(ok=(rc == 0), rc=rc, stdout=out, stderr=err, error=None if rc == 0 else err, t_ms=t_ms)
        if lang in ("rs", "rust", "rustlang"):
            p = td_path / "main.rs"
            p.write_text(code)
            # Compile without optimizations to reduce compile time on resource-constrained hosts
            rc1, out1, err1, _ = await _run(["rustc", str(p), "-o", str(td_path / "a.out")], None, timeout)
            if rc1 != 0:
                return ExecResp(ok=False, rc=rc1, stdout=out1, stderr=err1, error=err1, t_ms=0.0)
            rc, out, err, t_ms = await _run([str(td_path / "a.out")], None, timeout)
            return ExecResp(ok=(rc == 0), rc=rc, stdout=out, stderr=err, error=None if rc == 0 else err, t_ms=t_ms)
        if lang in ("asm", "nasm"):
            p = td_path / "prog.asm"
            p.write_text(code)
            rc1, out1, err1, _ = await _run(["nasm", "-felf64", str(p), "-o", str(td_path / "prog.o")], None, timeout)
            if rc1 != 0:
                return ExecResp(ok=False, rc=rc1, stdout=out1, stderr=err1, error=err1).model_dump() | {"t_ms": 0.0}
            rc2, out2, err2, _ = await _run(["ld", "-o", str(td_path / "a.out"), str(td_path / "prog.o")], None, timeout)
            if rc2 != 0:
                return ExecResp(ok=False, rc=rc2, stdout=out2, stderr=err2, error=err2, t_ms=0.0)
            rc, out, err, t_ms = await _run([str(td_path / "a.out")], None, timeout)
            return ExecResp(ok=(rc == 0), rc=rc, stdout=out, stderr=err, error=None if rc == 0 else err, t_ms=t_ms)

    raise HTTPException(status_code=400, detail=f"unsupported language: {lang}")

@app.post("/exec", response_model=None)  # <-- disable response filtering
async def exec_route(req: ExecReq):      # <-- no return type hint to ExecResp
    t0 = time.perf_counter()
    result = await exec_code(req)
    # Normalize to dict
    if isinstance(result, BaseModel):
        payload = result.model_dump()
    elif isinstance(result, dict):
        payload = dict(result)
    else:
        payload = jsonable_encoder(result)
    # Ensure t_ms is ALWAYS present
    payload.setdefault("t_ms", int((time.perf_counter() - t0) * 1000))
    return JSONResponse(payload)
