"""Record raw Bedrock Invoke and Mantle replies for the litefuzz behavior catalogs (LIVE-VALIDATION.md batches C and D).

Calls AWS directly, never through litellm. Each call writes request.json, response.json (status and every header)
and body.bin (raw bytes, for Invoke streams the raw event-stream frames) under --out/<call id>/, plus events.jsonl
with decoded frames for Invoke streams. C-O1 and D-cache stream the same cached system prompt twice, miss then hit.

    uv run python tests/live_validation/bedrock_messages_live.py --out /tmp/bedrock-live
    uv run python tests/live_validation/bedrock_messages_live.py --only C-O1 D-cache

SigV4 credentials come from the default botocore chain (AWS_PROFILE, env keys, SSO). Mantle calls sign with SigV4
too unless --mantle-api-key or BEDROCK_MANTLE_API_KEY is set, and D10 runs only with that key
"""

import argparse
import base64
import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final
from urllib.parse import quote

import botocore.session
import httpx
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials
from botocore.eventstream import EventStreamBuffer

ANTHROPIC_VERSION: Final = "bedrock-2023-05-31"
MANTLE_ANTHROPIC_VERSION: Final = "2023-06-01"
MANTLE_MESSAGES: Final = "/anthropic/v1/messages"
REDACTED_HEADERS: Final = frozenset({"authorization", "x-amz-security-token"})
CACHE_FILLER: Final = " ".join(
    f"Reference note {index}: the quick brown fox jumps over the lazy dog while the archivist files record {index}."
    for index in range(400)
)


@dataclass(frozen=True)
class Target:
    endpoint: str
    region: str
    sigv4: Credentials
    mantle_api_key: str | None


@dataclass(frozen=True)
class Call:
    call_id: str
    path: str
    body: dict[str, object]
    stream: bool = False
    extra_headers: dict[str, str] = field(default_factory=dict)
    auth: str = "sigv4"
    host: str = "runtime"


def _plain(max_tokens: int = 16, prompt: str = "Say hi") -> dict[str, object]:
    return {
        "anthropic_version": ANTHROPIC_VERSION,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }


def _cached_system() -> dict[str, object]:
    return {
        **_plain(),
        "system": [{"type": "text", "text": CACHE_FILLER, "cache_control": {"type": "ephemeral"}}],
    }


def _mantle(model: str, **extra: object) -> dict[str, object]:
    return {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": "Say hi"}], **extra}


def _mantle_cached(model: str) -> dict[str, object]:
    return _mantle(
        model,
        stream=True,
        system=[{"type": "text", "text": CACHE_FILLER, "cache_control": {"type": "ephemeral"}}],
    )


def build_mantle_calls(profile_id: str, base_id: str, has_api_key: bool) -> list[Call]:
    mantle_header: Final = {"anthropic-version": MANTLE_ANTHROPIC_VERSION}
    count: Final = f"{MANTLE_MESSAGES}/count_tokens"
    count_body: Final = {"model": base_id, "messages": [{"role": "user", "content": "hi"}]}

    def call(
        call_id: str,
        path: str,
        body: dict[str, object],
        extra_headers: dict[str, str] | None = None,
        auth: str = "sigv4",
    ) -> Call:
        headers: Final = {**mantle_header, **(extra_headers or {})}
        return Call(
            call_id, path, body, stream=body.get("stream") is True, extra_headers=headers, auth=auth, host="mantle"
        )

    calls: Final = [
        call("D1", MANTLE_MESSAGES, _mantle(profile_id)),
        call("D2", MANTLE_MESSAGES, _mantle(profile_id, stream=True)),
        call("D3", MANTLE_MESSAGES, _mantle(profile_id, tool_choice={"type": "auto"})),
        call(
            "D4", MANTLE_MESSAGES, _mantle(profile_id), extra_headers={"Authorization": "Bearer invalid"}, auth="none"
        ),
        call("D5", MANTLE_MESSAGES, _mantle("anthropic.claude-does-not-exist")),
        call("D6", MANTLE_MESSAGES, _mantle(profile_id, cache_control={"type": "ephemeral"})),
        call("D7", count, count_body),
        call("D8", count, {**count_body, "thinking": {"type": "enabled", "budget_tokens": 1024}}),
        call(
            "D9",
            MANTLE_MESSAGES,
            {
                **_mantle("anthropic.claude-haiku-4-5", max_tokens=8),
                "system": [{"type": "text", "text": "x", "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
            },
        ),
        call("D-cache-miss", MANTLE_MESSAGES, _mantle_cached(profile_id)),
        call("D-cache-hit", MANTLE_MESSAGES, _mantle_cached(profile_id)),
    ]
    if not has_api_key:
        return calls
    return [*calls, call("D10", MANTLE_MESSAGES, _mantle(profile_id, max_tokens=8), auth="joined-bearer")]


def _count_body() -> dict[str, object]:
    return {"input": {"converse": {"messages": [{"role": "user", "content": [{"text": "hi"}]}]}}}


def build_calls(profile_id: str, base_id: str, cross_region_only_id: str | None) -> list[Call]:
    invoke = f"/model/{quote(profile_id, safe='')}/invoke"
    stream = f"/model/{quote(profile_id, safe='')}/invoke-with-response-stream"
    calls = [
        Call("C1", invoke, _plain()),
        Call("C2", stream, _plain(prompt="Say café"), stream=True),
        Call(
            "C3",
            invoke,
            {**_plain(), "anthropic_beta": ["context-management-2025-06-27"]},
            extra_headers={"anthropic-beta": "context-management-2025-06-27"},
        ),
        Call("C4", invoke, {**_plain(), "compaction": {"type": "summarize"}, "anthropic_beta": ["compact-2026-09-04"]}),
        Call(
            "C5",
            invoke,
            {
                **_plain(),
                "context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]},
                "anthropic_beta": ["context-management-2025-06-27"],
            },
        ),
        Call("C6", invoke, {**_plain(), "cache_control": {"type": "ephemeral"}}),
        Call("C7", invoke, _plain(max_tokens=0)),
        Call("C8", invoke, _plain(), auth="fake-key"),
        Call("C9", invoke, _plain(), extra_headers={"Authorization": "Bearer invalid"}, auth="none"),
        Call("C10-base", f"/model/{quote(base_id, safe='')}/count-tokens", _count_body()),
        Call("C-O1-miss", stream, _cached_system(), stream=True),
        Call("C-O1-hit", stream, _cached_system(), stream=True),
    ]
    if cross_region_only_id:
        calls.append(
            Call("C10-cross-region-only", f"/model/{quote(cross_region_only_id, safe='')}/count-tokens", _count_body())
        )
    return calls


def _sigv4(headers: dict[str, str], url: str, payload: bytes, region: str, credentials: Credentials) -> dict[str, str]:
    request: Final = AWSRequest(method="POST", url=url, data=payload, headers=headers)
    SigV4Auth(credentials, "bedrock", region).add_auth(request)
    return dict(request.headers.items())


def _auth_headers(call: Call, url: str, payload: bytes, target: Target) -> dict[str, str]:
    headers: Final = {
        "content-type": "application/json",
        "accept": "application/json",
        "accept-encoding": "identity",
        **call.extra_headers,
    }
    match (call.auth, call.host == "mantle" and target.mantle_api_key is not None):
        case ("none", _):
            return headers
        case ("fake-key", _):
            fake: Final = Credentials("AKIAIOSFODNN7EXAMPLE", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")
            return _sigv4(headers, url, payload, target.region, fake)
        case ("joined-bearer", _):
            return {**headers, "authorization": f"Bearer junk, Bearer {target.mantle_api_key}"}
        case (_, True):
            return {**headers, "Authorization": f"Bearer {target.mantle_api_key}"}
        case _:
            return _sigv4(headers, url, payload, target.region, target.sigv4)


def _redacted(headers: dict[str, str]) -> dict[str, str]:
    return {name: ("<redacted>" if name.lower() in REDACTED_HEADERS else value) for name, value in headers.items()}


def _decoded_events(raw: bytes) -> list[dict[str, object]]:
    buffer = EventStreamBuffer()
    buffer.add_data(raw)
    events: list[dict[str, object]] = []
    for message in buffer:
        headers = {name: str(value) for name, value in message.headers.items()}
        payload = message.payload.decode("utf-8", errors="replace")
        event: dict[str, object] = {"headers": headers, "payload": payload}
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            events.append(event)
            continue
        chunk = parsed.get("bytes") if isinstance(parsed, dict) else None
        if isinstance(chunk, str):
            event["decoded_bytes"] = base64.b64decode(chunk).decode("utf-8", errors="replace")
        events.append(event)
    return events


def run_call(client: httpx.Client, call: Call, targets: dict[str, Target], out: Path) -> str:
    target_host: Final = targets[call.host]
    url = f"{target_host.endpoint}{call.path}"
    payload = json.dumps(call.body).encode()
    headers = _auth_headers(call, url, payload, target_host)
    target = out / call.call_id
    target.mkdir(parents=True, exist_ok=True)
    (target / "request.json").write_text(
        json.dumps({"method": "POST", "url": url, "headers": _redacted(headers), "body": call.body}, indent=2)
    )
    with client.stream("POST", url, content=payload, headers=headers) as response:
        raw = b"".join(response.iter_raw())
        (target / "response.json").write_text(
            json.dumps(
                {
                    "status": response.status_code,
                    "http_version": response.http_version,
                    "headers": response.headers.multi_items(),
                },
                indent=2,
            )
        )
    (target / "body.bin").write_bytes(raw)
    if call.stream and call.host == "runtime" and response.status_code == 200:
        (target / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in _decoded_events(raw)))
    return f"{call.call_id}: {response.status_code} ({len(raw)} bytes)"


def _matches(call_id: str, wanted: str) -> bool:
    return call_id == wanted or call_id.startswith(f"{wanted}-") or (len(wanted) == 1 and call_id.startswith(wanted))


def _selector(only: list[str] | None) -> Callable[[Call], bool]:
    def selected(call: Call) -> bool:
        return only is None or any(_matches(call.call_id, wanted) for wanted in only)

    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("bedrock-live"))
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--profile-id", default="us.anthropic.claude-haiku-5-5")
    parser.add_argument("--base-id", default="anthropic.claude-haiku-5-5")
    parser.add_argument("--cross-region-only-id", default=None)
    parser.add_argument("--mantle-api-key", default=os.environ.get("BEDROCK_MANTLE_API_KEY"))
    parser.add_argument("--cache-pause", type=float, default=2.0, help="seconds between the cache miss and hit")
    parser.add_argument("--only", nargs="*", default=None, help="call ids or prefixes, e.g. C1 C-O1 D")
    args = parser.parse_args()

    resolved = botocore.session.get_session().get_credentials()
    if resolved is None:
        raise SystemExit("no AWS credentials in the default chain")
    real = resolved.get_frozen_credentials()
    credentials = Credentials(real.access_key, real.secret_key, real.token)
    targets: Final = {
        "runtime": Target(f"https://bedrock-runtime.{args.region}.amazonaws.com", args.region, credentials, None),
        "mantle": Target(
            f"https://bedrock-mantle.{args.region}.api.aws", args.region, credentials, args.mantle_api_key
        ),
    }
    calls: Final = [
        *build_calls(args.profile_id, args.base_id, args.cross_region_only_id),
        *build_mantle_calls(args.profile_id, args.base_id, args.mantle_api_key is not None),
    ]
    with httpx.Client(timeout=120.0) as client:
        for call in filter(_selector(args.only), calls):
            if call.call_id.endswith("-hit"):
                time.sleep(args.cache_pause)
            sys.stdout.write(run_call(client, call, targets, args.out) + "\n")


if __name__ == "__main__":
    main()
