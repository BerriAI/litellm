"""Drives real proxy requests that make the detector open one prompt cache incident, next to two controls.

Each step waits for the detector's own Redis state to show the previous request was applied, because the
detector reads callbacks asynchronously and orders evidence by when each request started. Nothing here writes
alerts directly: the incident comes from the proxy.

  claude-demo-unproven  three misses on a prefix that was never cached: never assessed, no incident
  claude-demo-healthy   a proving write, a miss, a cache read, then two misses: the read resets, no incident
  claude-demo-east      a proving write, then three misses: one incident and one Slack post. Three more misses
                        find the same saved incident, so nothing new is opened or posted
"""

import json
import os
import sys
import time
import urllib.request
from collections.abc import Callable
from typing import Final

import redis

PROXY_URL: Final = os.environ["PROXY_URL"]
MASTER_KEY: Final = os.environ["LITELLM_MASTER_KEY"]
NOTIFIER_URL: Final = os.environ["NOTIFIER_URL"]
UI_URL: Final = os.environ["UI_URL"]
DEADLINE_SECONDS: Final = 60.0
POLL_SECONDS: Final = 0.2
STREAK_STATE: Final = "*prompt_cache_streak:*:state"
STORE: Final = redis.Redis(host=os.environ["REDIS_HOST"], port=6379, decode_responses=True)


def wait_for(what: str, condition: Callable[[], bool]) -> None:
    deadline: Final = time.monotonic() + DEADLINE_SECONDS
    while not condition():
        if time.monotonic() > deadline:
            sys.exit(f"timed out waiting for {what}")
        time.sleep(POLL_SECONDS)


def call(path: str, body: object | None = None) -> object:
    request: Final = urllib.request.Request(
        f"{PROXY_URL}{path}",
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {MASTER_KEY}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def send(model: str, scenario: str, turn: int) -> None:
    prefix: Final = f"You are the {model} demo assistant. " + "Answer briefly and cite the shared policy. " * 400
    call(
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": prefix, "cache_control": {"type": "ephemeral"}}],
                },
                {"role": "user", "content": f"Question {turn} [demo:{scenario}]"},
            ],
        },
    )


def streak_states() -> frozenset[str]:
    return frozenset(STORE.scan_iter(match=STREAK_STATE))


def epoch(state: str) -> str | None:
    return STORE.hget(state, "epoch")


def misses(state: str) -> int:
    return STORE.zcard(state.removesuffix(":state") + ":misses")


def prove(model: str) -> str:
    before: Final = streak_states()
    send(model, "write", 0)
    wait_for(f"{model} proof", lambda: len(streak_states() - before) == 1)
    (state,) = streak_states() - before
    return state


def miss(model: str, state: str, turn: int) -> None:
    previous: Final = (misses(state), epoch(state))
    send(model, "miss", turn)
    wait_for(f"{model} miss {turn}", lambda: (misses(state), epoch(state)) != previous)


def incidents() -> list[dict[str, object]]:
    page: Final = call("/alerts?type=prompt_cache&status=open")
    data: Final = page.get("data") if isinstance(page, dict) else None
    return [alert for alert in data if isinstance(alert, dict)] if isinstance(data, list) else []


def posts() -> int:
    with urllib.request.urlopen(f"{NOTIFIER_URL}/received", timeout=10) as response:
        count: Final = json.loads(response.read()).get("count")
    return count if isinstance(count, int) else 0


def main() -> None:
    if streak_states() or incidents():
        sys.exit("This stack already ran the demo. Run `demo.sh reset` first, then `demo.sh run`.")
    for turn in range(1, 4):
        send("claude-demo-unproven", "miss", turn)

    healthy: Final = prove("claude-demo-healthy")
    miss("claude-demo-healthy", healthy, 1)
    before_read: Final = epoch(healthy)
    send("claude-demo-healthy", "read", 2)
    wait_for("claude-demo-healthy read", lambda: epoch(healthy) != before_read)
    miss("claude-demo-healthy", healthy, 3)
    miss("claude-demo-healthy", healthy, 4)

    east: Final = prove("claude-demo-east")
    for turn in range(1, 4):
        miss("claude-demo-east", east, turn)
    wait_for(
        "the claude-demo-east incident", lambda: [alert["resource_id"] for alert in incidents()] == ["claude-demo-east"]
    )
    wait_for("one Slack post", lambda: posts() == 1)
    for turn in range(4, 7):
        miss("claude-demo-east", east, turn)

    opened: Final = sorted(str(alert["resource_id"]) for alert in incidents())
    if opened != ["claude-demo-east"] or posts() != 1:
        sys.exit(f"unexpected result: open incidents {opened}, Slack posts {posts()}")
    print(f"Open prompt cache incidents: {opened}. Slack posts: 1. Review them at {UI_URL}")


if __name__ == "__main__":
    main()
