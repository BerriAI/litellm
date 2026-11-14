"""Seed the pricing currency demo through the proxy's own management and chat APIs.

Safe to rerun: models are created only when their id is missing, the demo keys only when they do not exist, and each
synthetic request is recorded on the budget key's metadata right after it is answered, so a rerun resumes after the
last recorded request. Every request goes to the local mock upstream.
"""

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Final

PROXY: Final = os.environ.get("DEMO_PROXY_URL", "http://127.0.0.1:4000").rstrip("/")
MASTER_KEY: Final = os.environ["LITELLM_MASTER_KEY"]
BUDGET_KEY: Final = os.environ["DEMO_VIRTUAL_KEY"]
PRICING_KEY: Final = os.environ["DEMO_PRICING_KEY"]
MOCK_UPSTREAM: Final = {"api_base": "http://mock-upstream:8090/v1", "api_key": "mock-upstream-ignores-this"}
DEMO_GROUPS: Final = (
    "demo-currency-cheapest",
    "demo-currency-huf-model-info",
    "demo-currency-eur-mixed",
    "demo-currency-huf-db",
)
BUDGET_KEY_LIMIT_USD: Final = 0.03
SEED_PROGRESS: Final = "pricing_currency_demo_seeded_requests"
SECRET_PATTERN: Final = re.compile(r"sk-[A-Za-z0-9_-]{6,}")

Response = tuple[int, dict[str, object], dict[str, str]]

API_MODELS: Final[tuple[tuple[str, dict[str, object]], ...]] = (
    (
        "demo-eur-180",
        {
            "model_name": "demo-currency-cheapest",
            "litellm_params": {
                "model": "openai/demo-eur-180",
                **MOCK_UPSTREAM,
                "pricing_currency": "EUR",
                "usd_per_currency_unit": "1.20",
                "input_cost_per_token": 0.0000018,
                "output_cost_per_token": 0.0000036,
            },
            "model_info": {"id": "demo-eur-180"},
        },
    ),
    (
        "demo-eur-mixed",
        {
            "model_name": "demo-currency-eur-mixed",
            "litellm_params": {
                "model": "openai/demo-eur-mixed",
                **MOCK_UPSTREAM,
                "pricing_currency": "EUR",
                "usd_per_currency_unit": "1.20",
                "input_cost_per_token": 0.0000018,
            },
            "model_info": {"id": "demo-eur-mixed", "output_cost_per_token": 0.0000036},
        },
    ),
    (
        "demo-huf-db",
        {
            "model_name": "demo-currency-huf-db",
            "litellm_params": {"model": "openai/demo-huf-db-reported-cost", **MOCK_UPSTREAM},
            "model_info": {
                "id": "demo-huf-db",
                "pricing_currency": "HUF",
                "usd_per_currency_unit": "0.003",
                "input_cost_per_token": 0.00072,
                "output_cost_per_token": 0.00144,
            },
        },
    ),
)

SYNTHETIC_TRAFFIC: Final = (
    "demo-currency-cheapest",
    "demo-currency-cheapest",
    "demo-currency-huf-model-info",
    "demo-currency-huf-model-info",
    "demo-currency-eur-mixed",
    "demo-currency-eur-mixed",
)


def _redacted(text: str) -> str:
    return SECRET_PATTERN.sub("sk-[redacted]", text)[:300]


def _call(method: str, path: str, key: str, body: object | None = None) -> Response:
    request: Final = urllib.request.Request(
        f"{PROXY}{path}",
        method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read() or b"{}"), dict(response.headers.items())
    except urllib.error.HTTPError as error:
        payload: Final = error.read()
        try:
            parsed: Final = json.loads(payload or b"{}")
        except json.JSONDecodeError:
            return error.code, {"raw": payload.decode(errors="replace")}, {}
        return error.code, parsed if isinstance(parsed, dict) else {"raw": parsed}, {}
    except urllib.error.URLError as error:
        return 0, {"raw": str(error.reason)}, {}


def _fail(what: str, status: int, body: dict[str, object]) -> None:
    sys.exit(f"{what} failed with HTTP {status}: {_redacted(json.dumps(body))}")


def _wait_until(description: str, ready: Callable[[], bool], attempts: int = 120) -> None:
    for _ in range(attempts):
        if ready():
            return
        time.sleep(1)
    sys.exit(f"Timed out waiting for {description}")


def _listed(path: str) -> bool:
    status, body, _ = _call("GET", path, MASTER_KEY)
    data: Final = body.get("data")
    return status == 200 and isinstance(data, list) and len(data) > 0


def _ensure_models() -> None:
    for model_id, spec in API_MODELS:
        if _listed(f"/v2/model/info?modelId={model_id}"):
            print(f"model {model_id}: already exists")
            continue
        status, body, _ = _call("POST", "/model/new", MASTER_KEY, spec)
        if status != 200:
            _fail(f"creating model {model_id}", status, body)
        print(f"model {model_id}: created")


def _key_info(key: str) -> dict[str, object] | None:
    status, body, _ = _call("GET", f"/key/info?key={key}", MASTER_KEY)
    info: Final = body.get("info")
    return info if status == 200 and isinstance(info, dict) else None


def _ensure_key(key: str, alias: str, max_budget: float | None) -> dict[str, object]:
    existing: Final = _key_info(key)
    if existing is not None:
        print(f"key {alias}: already exists")
        return existing
    status, body, _ = _call(
        "POST",
        "/key/generate",
        MASTER_KEY,
        {
            "key": key,
            "key_alias": alias,
            "models": list(DEMO_GROUPS),
            "max_budget": max_budget,
            "metadata": {"demo": "pricing-currency"},
        },
    )
    if status != 200:
        _fail(f"creating key {alias}", status, body)
    print(f"key {alias}: created" + ("" if max_budget is None else f" with a ${max_budget} USD budget"))
    created: Final = _key_info(key)
    if created is None:
        sys.exit(f"key {alias}: created but not readable")
    return created


def _seeded_requests(key_info: dict[str, object]) -> int:
    metadata: Final = key_info.get("metadata")
    progress: Final = metadata.get(SEED_PROGRESS) if isinstance(metadata, dict) else None
    return progress if isinstance(progress, int) else 0


def _record_progress(done: int) -> None:
    status, body, _ = _call(
        "POST",
        "/key/update",
        MASTER_KEY,
        {"key": BUDGET_KEY, "metadata": {"demo": "pricing-currency", SEED_PROGRESS: done}},
    )
    if status != 200:
        _fail("recording seed progress", status, body)


def _send_traffic(already_sent: int) -> None:
    for index, group in enumerate(SYNTHETIC_TRAFFIC):
        if index < already_sent:
            continue
        status, body, headers = _call(
            "POST",
            "/v1/chat/completions",
            BUDGET_KEY,
            {
                "model": group,
                "messages": [{"role": "user", "content": "Synthetic demo request, not live model evidence."}],
                "user": "synthetic-demo",
                "metadata": {"tags": ["synthetic-demo"]},
            },
        )
        if status != 200:
            _fail(f"synthetic request {index + 1} to {group}", status, body)
        _record_progress(index + 1)
        print(
            f"{group}: served by {headers.get('x-litellm-model-id')} for "
            f"${headers.get('x-litellm-response-cost')} USD"
        )


def main() -> None:
    _wait_until("the proxy", lambda: _call("GET", "/health/readiness", MASTER_KEY)[0] == 200)
    _ensure_models()
    budget_key: Final = _ensure_key(BUDGET_KEY, "demo-currency-budget-key", BUDGET_KEY_LIMIT_USD)
    _ensure_key(PRICING_KEY, "demo-currency-pricing-key", None)
    already_sent: Final = _seeded_requests(budget_key)
    if already_sent >= len(SYNTHETIC_TRAFFIC):
        print("synthetic traffic: already sent on an earlier run, skipping")
        return
    for group in DEMO_GROUPS:
        _wait_until(f"the proxy to list {group}", lambda group=group: _listed(f"/model_group/info?model_group={group}"))
    _send_traffic(already_sent)
    print("Seed complete. Spend rows appear on the Logs page within about a minute.")


if __name__ == "__main__":
    main()
