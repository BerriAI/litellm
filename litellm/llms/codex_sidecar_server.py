"""Codex CLI sidecar exposing an OpenAI-like HTTP endpoint.

This module mirrors the reference implementation from the Codex hybrid
proposal.  It keeps the Codex CLI as the only component that talks to the
remote service, while exposing a stable local HTTP API for LiteLLM.

The server is intentionally lightweight and is started on-demand by the
Codex agent provider if no ``CODEX_AGENT_API_BASE`` is configured.  The
default host/port can be overridden via ``CODEX_SIDECAR_HOST`` and
``CODEX_SIDECAR_PORT`` environment variables.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import signal
import sys
import time
from dataclasses import dataclass
from statistics import mean
from typing import AsyncIterator, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class _Settings:
    codex_cmd: List[str]
    input_mode: str
    prompt_flag: str
    max_concurrency: int
    abs_timeout_s: float
    idle_timeout_s: float
    retries: int
    log_level: str

    @staticmethod
    def load() -> "_Settings":
        codex_cmd_str = os.getenv("CODEX_CMD", "codex exec --json --stream")
        input_mode = os.getenv("CODEX_INPUT_MODE", "prompt").strip().lower()
        prompt_flag = os.getenv("CODEX_PROMPT_FLAG", "--prompt")
        max_concurrency = int(os.getenv("MAX_CONCURRENCY", "4"))
        abs_timeout_s = float(os.getenv("ABS_TIMEOUT_S", "90"))
        idle_timeout_s = float(os.getenv("IDLE_TIMEOUT_S", "25"))
        retries = int(os.getenv("RETRIES", "1"))
        log_level = os.getenv("LOG_LEVEL", "info").lower()
        return _Settings(
            codex_cmd=shlex.split(codex_cmd_str),
            input_mode=input_mode,
            prompt_flag=prompt_flag,
            max_concurrency=max_concurrency,
            abs_timeout_s=abs_timeout_s,
            idle_timeout_s=idle_timeout_s,
            retries=retries,
            log_level=log_level,
        )


SETTINGS = _Settings.load()


def _vlog(*args) -> None:
    if SETTINGS.log_level == "debug":
        print("[DEBUG]", *args, file=sys.stderr)


def _ilog(*args) -> None:
    print("[INFO]", *args, file=sys.stderr)


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="Model name (e.g., gpt-5).")
    messages: List[ChatMessage] = Field(..., description="OpenAI-style messages array.")
    stream: bool = Field(default=False)


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------


class ProcessError(Exception):
    def __init__(self, code: int, stderr: str):
        super().__init__(f"Process exited with code {code}: {stderr[:2000]}")
        self.code = code
        self.stderr = stderr


async def _kill_process_tree(proc: asyncio.subprocess.Process) -> None:
    try:
        if sys.platform != "win32":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except ProcessLookupError:
        pass
    except Exception as exc:  # pragma: no cover - best effort cleanup
        _vlog("kill error", exc)


async def _read_stream_lines(
    stream: asyncio.StreamReader, idle_timeout: float
) -> AsyncIterator[str]:
    while True:
        try:
            raw = await asyncio.wait_for(stream.readline(), timeout=idle_timeout)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(f"No output for {idle_timeout}s") from exc
        if not raw:
            break
        try:
            line = raw.decode("utf-8", errors="ignore")
        except Exception:
            line = raw.decode("utf-8", errors="ignore") if isinstance(raw, (bytes, bytearray)) else str(raw)
        yield line.rstrip("\r\n")


async def _run_codex_once(
    prompt: str,
    model: str,
    input_mode: str,
    args: List[str],
    prompt_flag: str,
    abs_timeout_s: float,
    idle_timeout_s: float,
) -> Tuple[int, str, str]:
    argv = list(args)
    stdin_data: Optional[bytes] = None

    if input_mode == "flag":
        argv += [prompt_flag, prompt]
    elif input_mode == "json":
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}
        stdin_data = (json.dumps(payload) + "\n").encode("utf-8")
    else:
        stdin_data = (prompt + "\n").encode("utf-8")

    _vlog("spawning", argv)

    preexec = os.setsid if sys.platform != "win32" else None
    creationflags = 0x00000200 if sys.platform == "win32" else 0

    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        preexec_fn=preexec if sys.platform != "win32" else None,
        creationflags=creationflags,
    )

    async def _writer() -> None:
        if stdin_data is not None and proc.stdin:
            try:
                proc.stdin.write(stdin_data)
                await proc.stdin.drain()
            except Exception:
                pass
        if proc.stdin:
            try:
                proc.stdin.close()
            except Exception:
                pass

    async def _reader() -> Tuple[str, str]:
        stdout_chunks: List[str] = []
        stderr_chunks: List[str] = []
        try:
            async for line in _read_stream_lines(proc.stdout, idle_timeout_s):
                stdout_chunks.append(line)
            async for line in _read_stream_lines(proc.stderr, idle_timeout_s):
                stderr_chunks.append(line)
        except TimeoutError as exc:
            await _kill_process_tree(proc)
            raise exc
        return "\n".join(stdout_chunks), "\n".join(stderr_chunks)

    try:
        writer_task = asyncio.create_task(_writer())
        reader_task = asyncio.create_task(_reader())
        done, pending = await asyncio.wait(
            {writer_task, reader_task},
            timeout=abs_timeout_s,
            return_when=asyncio.ALL_COMPLETED,
        )
        if pending:
            await _kill_process_tree(proc)
            for p in pending:
                p.cancel()
            raise TimeoutError(f"Absolute timeout after {abs_timeout_s}s")
        stdout_text, stderr_text = await reader_task
        code = await proc.wait()
        return code, stdout_text, stderr_text
    finally:  # pragma: no cover - cleanup path
        if proc.returncode is None:
            await _kill_process_tree(proc)


def _messages_to_prompt(messages: List[ChatMessage]) -> str:
    parts: List[str] = []
    for m in messages:
        role = (m.role or "").strip().lower()
        content = m.content or ""
        if role in ("system", "user"):
            parts.append(content)
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(content)
    return "\n".join(parts).strip()


def _is_transient_exit(code: int) -> bool:
    return code in (70, 75)


_SEM = asyncio.Semaphore(SETTINGS.max_concurrency)


# ---------------------------------------------------------------------------
# HTTP app
# ---------------------------------------------------------------------------


app = FastAPI(title="Codex Sidecar", version="0.1.0")


@app.get("/healthz")
async def healthz() -> Dict[str, object]:
    return {"ok": True, "concurrency": SETTINGS.max_concurrency}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    prompt = _messages_to_prompt(req.messages)
    if not prompt:
        raise HTTPException(status_code=400, detail="Empty prompt")

    async def _invoke_nonstream() -> JSONResponse:
        async with _SEM:
            retries = SETTINGS.retries
            attempt = 0
            last_err: Optional[str] = None
            while True:
                attempt += 1
                try:
                    code, stdout_text, stderr_text = await _run_codex_once(
                        prompt=prompt,
                        model=req.model,
                        input_mode=SETTINGS.input_mode,
                        args=SETTINGS.codex_cmd,
                        prompt_flag=SETTINGS.prompt_flag,
                        abs_timeout_s=SETTINGS.abs_timeout_s,
                        idle_timeout_s=SETTINGS.idle_timeout_s,
                    )
                    if code != 0:
                        if _is_transient_exit(code) and attempt <= retries + 1:
                            _ilog(f"Transient exit ({code}). retry {attempt}/{retries}")
                            await asyncio.sleep(min(1.0 * attempt, 3.0))
                            continue
                        raise ProcessError(code, stderr_text)
                    content = stdout_text.strip()
                    body = {
                        "id": "chatcmpl-sidecar",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": req.model,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": content or None},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                    return JSONResponse(body)
                except TimeoutError as exc:
                    last_err = str(exc)
                    if attempt <= retries + 1:
                        _ilog(f"Timeout, retry {attempt}/{retries}")
                        continue
                    raise HTTPException(status_code=504, detail=f"Timeout: {last_err}")
                except ProcessError as exc:
                    raise HTTPException(status_code=502, detail=f"Codex error ({exc.code}): {exc.stderr[:800]}")
                except Exception as exc:  # noqa: BLE001
                    last_err = str(exc)
                    raise HTTPException(status_code=500, detail=f"Unexpected: {last_err[:800]}")

    async def _invoke_stream() -> StreamingResponse:
        async def gen() -> AsyncIterator[bytes]:
            async with _SEM:
                retries = SETTINGS.retries
                attempt = 0
                while True:
                    attempt += 1
                    try:
                        code, stdout_text, stderr_text = await _run_codex_once(
                            prompt=prompt,
                            model=req.model,
                            input_mode=SETTINGS.input_mode,
                            args=SETTINGS.codex_cmd,
                            prompt_flag=SETTINGS.prompt_flag,
                            abs_timeout_s=SETTINGS.abs_timeout_s,
                            idle_timeout_s=SETTINGS.idle_timeout_s,
                        )
                        if code != 0:
                            if _is_transient_exit(code) and attempt <= retries + 1:
                                _ilog(f"Transient exit ({code}). retry {attempt}/{retries}")
                                await asyncio.sleep(min(1.0 * attempt, 3.0))
                                continue
                            err = {"error": {"message": f"Codex error ({code})"}}
                            yield f"data: {json.dumps(err)}\n\n".encode("utf-8")
                            return
                        for line in stdout_text.splitlines():
                            if not line:
                                continue
                            chunk = {
                                "id": "chatcmpl-sidecar",
                                "object": "chat.completion.chunk",
                                "model": req.model,
                                "choices": [{"index": 0, "delta": {"content": line + "\n"}}],
                            }
                            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8")
                        yield b"data: [DONE]\n\n"
                        return
                    except TimeoutError as exc:
                        if attempt <= retries + 1:
                            _ilog(f"Timeout, retry {attempt}/{retries}")
                            await asyncio.sleep(min(1.0 * attempt, 3.0))
                            continue
                        err = {"error": {"message": f"Timeout: {str(exc)}"}}
                        yield f"data: {json.dumps(err)}\n\n".encode("utf-8")
                        return
                    except Exception as exc:  # noqa: BLE001
                        err = {"error": {"message": f"Unexpected: {str(exc)}"}}
                        yield f"data: {json.dumps(err)}\n\n".encode("utf-8")
                        return

        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    if not req.stream:
        return await _invoke_nonstream()
    return await _invoke_stream()


# ---------------------------------------------------------------------------
# Entrypoint helpers
# ---------------------------------------------------------------------------


def serve(host: str = "127.0.0.1", port: int = 8077) -> None:
    """Run the sidecar using uvicorn."""

    _ilog("Codex sidecar starting with config:")
    _ilog(f"  CODEX_CMD        = {SETTINGS.codex_cmd}")
    _ilog(f"  CODEX_INPUT_MODE = {SETTINGS.input_mode}")
    _ilog(f"  MAX_CONCURRENCY  = {SETTINGS.max_concurrency}")
    _ilog(f"  ABS_TIMEOUT_S    = {SETTINGS.abs_timeout_s}")
    _ilog(f"  IDLE_TIMEOUT_S   = {SETTINGS.idle_timeout_s}")
    _ilog(f"  RETRIES          = {SETTINGS.retries}")
    _ilog(f"  LOG_LEVEL        = {SETTINGS.log_level}")

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - runtime requirement
        print("uvicorn is required to run the Codex sidecar", file=sys.stderr)
        raise exc

    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":  # pragma: no cover - manual invocation only
    host = os.getenv("CODEX_SIDECAR_HOST", "127.0.0.1")
    port = int(os.getenv("CODEX_SIDECAR_PORT", "8077"))
    serve(host, port)

