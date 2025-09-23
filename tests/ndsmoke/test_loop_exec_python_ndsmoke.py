import asyncio
import os
import re
import socket
import sys
import time
from typing import Dict, Any, List, Optional

import httpx
import pytest
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Live, non-deterministic smoke. Requires a local model (e.g., Ollama) configured via .env
# and will skip unless DOCKER_MINI_AGENT=1 is set to signal a live local environment.

TIMEOUT_TOTAL = int(os.getenv("NDSMOKE_TIMEOUT", "300"))  # ~5 minutes
TOOL_TIMEOUT = int(os.getenv("NDSMOKE_TOOL_TIMEOUT", "60"))
MAX_ITERS = int(os.getenv("NDSMOKE_MAX_ITERS", "4"))

CODE_RE = re.compile(r"```(?:python)?\s*(?P<code>[\s\S]*?)```", re.IGNORECASE)


def _can_connect(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def extract_code_block(text: str) -> Optional[str]:
    m = CODE_RE.search(text or "")
    return m.group("code").strip() if m else None


async def llm_chat(model: str, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    import litellm
    return await litellm.acompletion(model=model, messages=messages)


async def run_python(code: str, timeout: float = 60.0) -> Dict[str, Any]:
    import tempfile
    from asyncio.subprocess import PIPE

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, path, stdout=PIPE, stderr=PIPE
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return {"ok": False, "rc": -1, "stdout": "", "stderr": f"timeout after {timeout}s"}
        rc = proc.returncode or 0
        return {
            "ok": rc == 0,
            "rc": rc,
            "stdout": (out or b"").decode("utf-8", "replace"),
            "stderr": (err or b"").decode("utf-8", "replace"),
        }
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass


@pytest.mark.ndsmoke
def test_loop_exec_python_ndsmoke():
    """
    Explicit loop smoke (generate -> run -> observe -> repair -> repeat).
    Ensures the model iterates until the script executes cleanly (rc==0, no stderr)
    and prints the two expected outputs.
    """
    if os.getenv("DOCKER_MINI_AGENT", "0") != "1":
        pytest.skip("DOCKER_MINI_AGENT not set; skipping live ndsmoke")

    # Optional API reachability guard when using local gateway; not strictly required here
    api_host = os.getenv("MINI_AGENT_API_HOST", "127.0.0.1")
    api_port = int(os.getenv("MINI_AGENT_API_PORT", "8788"))
    if not _can_connect(api_host, api_port):
        pytest.skip(f"mini-agent API not reachable on {api_host}:{api_port}")

    model = os.getenv("LITELLM_DEFAULT_CODE_MODEL", "ollama/glm4:latest")

    user_prompt = (
        "Implement a Python function `compress_runs(s: str) -> str` that compresses runs of repeated characters "
        "like aaabbc -> a3b2c1. Return only a single Python code block and, in __main__, print two tests: "
        "compress_runs('aaabbc') and compress_runs('aabbccddeeffgg')."
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "You return only one Python code block that runs as-is."},
        {"role": "user", "content": user_prompt},
    ]

    t0 = time.perf_counter()
    success = False
    stdout_accum = ""

    for _ in range(MAX_ITERS):
        # ask model
        resp = asyncio.run(llm_chat(model, messages))
        msg = resp["choices"][0]["message"]
        content = msg.get("content") or ""
        code = extract_code_block(content)
        if not code:
            messages.append({
                "role": "assistant",
                "content": "No code block found. Return ONLY a Python code block with the full script.",
            })
            continue
        # run
        inv = asyncio.run(run_python(code, timeout=TOOL_TIMEOUT))
        stdout_accum += inv.get("stdout", "")
        if inv.get("ok") and (inv.get("stderr") or "").strip() == "":
            success = True
            break
        # add observation + directive
        preview = {
            "rc": inv.get("rc"),
            "stdout_tail": inv.get("stdout", "")[-500:],
            "stderr_tail": inv.get("stderr", "")[-1500:],
        }
        directive = (
            "Observation from last run (preview):\n" + str(preview) +
            "\n\nFix the code and return ONLY a single Python code block."
        )
        messages.append({"role": "assistant", "content": directive})
        if TIMEOUT_TOTAL and (time.perf_counter() - t0) > TIMEOUT_TOTAL:
            break

    assert success, "Script did not run successfully within iteration/time budget"
    low = stdout_accum.lower()
    assert "a3b2c1" in low and "a2b2c2d2e2f2g2" in low
