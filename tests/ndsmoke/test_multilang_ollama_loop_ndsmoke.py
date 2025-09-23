import os, re, sys, socket, json, asyncio, httpx, pytest
from typing import Dict, Any, List, Optional

TIMEOUT_TOTAL = int(os.getenv("NDSMOKE_TIMEOUT", "300"))  # ~5 minutes
TOOL_TIMEOUT = int(os.getenv("NDSMOKE_TOOL_TIMEOUT", "90"))
MAX_ITERS = int(os.getenv("NDSMOKE_MAX_ITERS", "4"))

CODE_RE = re.compile(r"```(?P<lang>[a-z0-9_+-]+)?\s*(?P<code>[\s\S]*?)```", re.IGNORECASE)


def _can_connect(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


async def llm_chat(model: str, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    import litellm
    return await litellm.acompletion(model=model, messages=messages)


async def exec_rpc(language: str, code: str, timeout: float = 90.0) -> Dict[str, Any]:
    url = os.getenv('MINI_AGENT_EXEC_BASE','http://127.0.0.1:8790').rstrip('/') + '/exec'
    async with httpx.AsyncClient(timeout=timeout+5) as client:
        r = await client.post(url, json={"language": language, "code": code, "timeout_sec": timeout})
        r.raise_for_status()
        return r.json()


def _extract(text: str) -> Optional[Dict[str,str]]:
    m = CODE_RE.search(text or "")
    if not m:
        return None
    return {"lang": (m.group("lang") or "").strip(), "code": m.group("code").strip()}


async def _loop_lang(model: str, lang: str, prompt: str, marker: str) -> Dict[str, Any]:
    messages = [
        {"role":"system","content":"Return only one code block for the requested language."},
        {"role":"user","content": prompt},
    ]
    import time
    t0=time.perf_counter()
    last_inv: Dict[str, Any] = {}
    for _ in range(MAX_ITERS):
        try:
            resp = await llm_chat(model, messages)
        except Exception as e:
            messages.append({"role":"assistant","content": f"Observation: model error {type(e).__name__}: {str(e)[:200]}\nReturn ONLY one code block for %s." % (lang,)})
            if (time.perf_counter()-t0) > TIMEOUT_TOTAL:
                break
            continue
        msg = resp["choices"][0]["message"]
        content = msg.get("content") or ""
        blk = _extract(content)
        if not blk:
            messages.append({"role":"assistant","content":"No code block found. Return ONLY one code block."})
            continue
        language = (blk["lang"] or lang).split()[0]
        inv = await exec_rpc(language, blk["code"], timeout=TOOL_TIMEOUT)
        last_inv = inv
        if inv.get("ok") and (inv.get("stderr") or "").strip() == "" and marker in (inv.get("stdout") or ""):
            return {"ok": True, "last": inv}
        preview = {
            "rc": inv.get("rc"),
            "stdout_tail": (inv.get("stdout") or "")[-500:],
            "stderr_tail": (inv.get("stderr") or "")[-1500:]
        }
        messages.append({"role":"assistant","content": f"Observation: {json.dumps(preview)}\nFix and return ONLY one code block."})
        if (time.perf_counter()-t0) > TIMEOUT_TOTAL:
            break
    return {"ok": False, "last": last_inv}


@pytest.mark.ndsmoke
@pytest.mark.live_ollama
@pytest.mark.parametrize("lang, marker, prompt", [
    ("ts", "TS_OK", "Return only one ```ts code block that runs with ts-node. Write function runLengthEncode(s: string): string and in main print two tests and then print TS_OK."),
    ("c", "C_OK", "Return only one ```c code block. Program prints C_OK to stdout. Compile with -Wall -Werror; fix if it fails."),
    ("cpp", "CPP_OK", "Return only one ```cpp code block (C++20). Program prints CPP_OK. Compile with -O2 -std=c++20 -Wall -Werror; fix if it fails."),
    ("go", "GO_OK", "Return only one ```go code block. Program prints GO_OK. If it fails to run, fix and retry."),
    ("java", "JAVA_OK", "Return only one ```java code block. Class Main prints JAVA_OK. If javac/java fails, fix and retry."),
    ("rs", "RUST_OK", "Return only one ```rust code block. Program prints RUST_OK. If rustc fails, fix and retry."),
    ("asm", "ASM_OK", "Return only one ```asm code block (NASM x86_64 Linux). Program prints ASM_OK to stdout using sys_write syscall. If assembly/link fails, fix and retry."),
])
@pytest.mark.timeout(400)
def test_multilang_loop_ndsmoke(lang: str, marker: str, prompt: str):
    if os.getenv('DOCKER_MINI_AGENT','0') != '1':
        pytest.skip('DOCKER_MINI_AGENT not set; skipping live docker ndsmoke')
    if not _can_connect('127.0.0.1', 8790):
        pytest.skip('exec RPC not reachable')
    model = os.getenv('LITELLM_DEFAULT_CODE_MODEL','ollama/glm4:latest')
    out = asyncio.run(_loop_lang(model, lang, prompt, marker))
    assert out.get("ok") is True, f"final invocation: {out.get('last')}"
