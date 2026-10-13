from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Final

from common import (
    ARTIFACTS,
    BASE_URL,
    MODEL,
    ROOT,
    array_value,
    gateway,
    load,
    now,
    object_value,
    save,
    string_value,
)


def run_session(session: dict[str, object]) -> None:
    sid: Final = str(session["session_id"])
    directory: Final = ROOT / "scratch" / sid
    directory.mkdir(parents=True, exist_ok=True)
    environment: Final = {
        **{k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC_", "CLAUDE_", "LITELLM_"))},
        "ANTHROPIC_BASE_URL": BASE_URL,
        "ANTHROPIC_API_KEY": os.environ["LITELLM_API_KEY"],
        "ANTHROPIC_AUTH_TOKEN": os.environ["LITELLM_API_KEY"],
        "ANTHROPIC_CUSTOM_HEADERS": f"x-litellm-session-id: {sid}\nx-litellm-tags: semantic-spend-demo",
        "CLAUDE_CONFIG_DIR": str(directory / "claude-config"),
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    }
    prompts: Final = session["prompts"]
    if not isinstance(prompts, list):
        raise ValueError("Session prompts must be a list")
    for index, prompt in enumerate(prompts):
        output: Final = ARTIFACTS / "traffic" / f"{sid}-{index + 1:02d}.json"
        if output.exists():
            continue
        command: Final = [
            "claude",
            "--bare",
            "-p",
            str(prompt),
            "--model",
            MODEL,
            "--tools",
            "",
            "--strict-mcp-config",
            "--mcp-config",
            '{"mcpServers":{}}',
            "--setting-sources",
            "",
            "--output-format",
            "json",
            "--max-turns",
            "1",
            "--effort",
            "low",
            "--system-prompt",
            "You are an engineering assistant for synthetic task-spend demonstrations. "
            "Answer in at most 120 words, with concise illustrative code where useful. "
            "You have no tools and must not claim to execute commands or modify files. "
            "Never request or include actual credentials.",
            *(["--session-id", sid] if index == 0 else ["--resume", sid]),
        ]
        result: Final = subprocess.run(
            command,
            cwd=directory,
            env=environment,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=120,
        )
        data: Final = object_value(json.loads(result.stdout))
        if result.returncode or data.get("is_error"):
            save(output.with_suffix(".error.json"), {"exit": result.returncode, "is_error": data.get("is_error")})
            raise RuntimeError(f"Claude failed for {sid} prompt {index + 1}")
        save(
            output,
            {
                "session_id": sid,
                "turn_index": index + 1,
                "prompt": prompt,
                "result": data.get("result"),
                "claude_session_id": data.get("session_id"),
                "client_cost_estimate": data.get("total_cost_usd"),
                "usage": data.get("usage"),
                "completed_at": now(),
                "exit_code": result.returncode,
            },
        )
        print(f"{session['name']} {index + 1}/{len(prompts)} completed", flush=True)


def main() -> None:
    if shutil.disk_usage(ROOT).free < 50 * 1024**3:
        raise RuntimeError("Less than 50 GiB free")
    manifest_path: Final = ARTIFACTS / "manifest.json"
    if not manifest_path.exists():
        definitions: Final = array_value(load(ROOT / "demo.json"))
        user: Final = string_value(object_value(gateway().get("/key/info")["info"])["user_id"])
        sessions: Final = tuple(
            {**object_value(definition), "session_id": str(uuid.uuid4())} for definition in definitions
        )
        save(
            manifest_path,
            {
                "gateway": BASE_URL,
                "user_id": user,
                "start": now(),
                "end": None,
                "model": MODEL,
                "sessions": sessions,
                "synthetic": True,
            },
        )
    manifest: Final = object_value(load(manifest_path))
    with ThreadPoolExecutor(max_workers=3) as executor:
        tuple(executor.map(run_session, (object_value(s) for s in array_value(manifest["sessions"]))))
    if not manifest.get("end"):
        save(manifest_path, {**manifest, "end": now()})
    print("All demo prompts completed", flush=True)


if __name__ == "__main__":
    main()
