#!/usr/bin/env python3
"""Measure /chat/completions latency against the LiteLLM proxy."""

import argparse
import asyncio
import random
import time

import httpx

"""
What makes this file be capable of reproducing the latency issue when using prometheus callbacks:
1. One HTTP client per user, and it sends requests as fast as possible.
2. Don't send too many requests at once but don't be too slow either.
3. Passing a user_id to call makes the issue more reproducible.
4. Sending the requests in waves makes the issue more reproducible.

USER_MODES:
- random: each user gets a random 10-digit ID (default, most reproducible for Prometheus)
- sequential: each user gets sequential ID (1, 2, 3, ...) - fewer cache misses
- none: no user field in payload - skips end_user lookup entirely
- created: create end users via POST /customer/new before the test, then use those IDs (requires --key-mode per_user)

KEY_MODES (add-on; can be combined with any user mode):
- shared: all users share the same API key (default)
- per_user: create one key per user via /key/generate before the test; each user uses their own key
  (required when --user-mode created)
"""

# -----------------------------------------------------------------------------
# Constants (edit as needed)
# -----------------------------------------------------------------------------
BASE_URL = "http://localhost:4000"
API_KEY = "sk-1234"
MODEL = "gpt-o1"  # must match a model_name in your proxy config (e.g. test_config_123.yaml)
NUM_REQUESTS = 1000
NUM_CONCURRENT = 100  # Concurrent users; each has one connection, reuses it for their requests
MESSAGES = [{"role": "user", "content": "Say hello in one word."}]
TIMEOUT = 30000.0
# -----------------------------------------------------------------------------

_results: list[tuple[float, bool]] = []  # shared; asyncio is single-threaded so no lock needed


async def create_keys_for_users(
    base_url: str, master_key: str, num_keys: int
) -> list[str]:
    """Create num_keys via POST /key/generate; return list of keys."""
    path = f"{base_url.rstrip('/')}/key/generate"
    headers = {
        "Authorization": f"Bearer {master_key}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:

        async def create_one(_: int) -> str:
            resp = await client.post(path, json={}, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            key = data.get("key")
            if not key:
                raise ValueError(f"Key generation response missing 'key': {data}")
            return key

        tasks = [create_one(i) for i in range(num_keys)]
        return list(await asyncio.gather(*tasks))


async def create_end_users_for_test(
    base_url: str, master_key: str, user_ids: list[str]
) -> list[str]:
    """Create end users via POST /customer/new; return list of user_ids (as created)."""
    path = f"{base_url.rstrip('/')}/customer/new"
    headers = {
        "Authorization": f"Bearer {master_key}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:

        async def create_one(uid: str) -> str:
            resp = await client.post(
                path, json={"user_id": uid}, headers=headers
            )
            resp.raise_for_status()
            return uid

        tasks = [create_one(uid) for uid in user_ids]
        return list(await asyncio.gather(*tasks))


async def run_user(
    user_id: int,
    num_requests: int,
    base_url: str,
    path: str,
    payload: dict,
    headers: dict,
) -> None:
    """One user: one client, fires requests as fast as possible."""
    async with httpx.AsyncClient(base_url=base_url, timeout=TIMEOUT) as client:
        for req_num in range(1, num_requests + 1):
            start = time.perf_counter()
            try:
                resp = await client.post(path, json=payload, headers=headers)
                resp.read()  # consume full body - timer stops after last byte received
                ok = resp.is_success
            except Exception:
                ok = False
            lat = time.perf_counter() - start
            _results.append((lat, ok))
            status = "[OK]" if ok else "[FAIL]"
            print(f"  {status} User {user_id} request {req_num:3d}: {lat:.3f}s", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure /chat/completions latency against the LiteLLM proxy."
    )
    parser.add_argument(
        "--user-mode",
        choices=["random", "sequential", "none", "created"],
        required=True,
        help="random: random user IDs; sequential: 1,2,3,...; "
        "none: no user in payload; created: create end users via API (requires --key-mode per_user)",
    )
    parser.add_argument(
        "--key-mode",
        choices=["shared", "per_user"],
        required=True,
        help="shared: all users use same API key; per_user: create one key per user",
    )
    args = parser.parse_args()
    if args.user_mode == "created" and args.key_mode != "per_user":
        parser.error("--user-mode created requires --key-mode per_user")
    return args


async def main(
    user_mode: str = "random", key_mode: str = "shared"
) -> None:
    base_url = BASE_URL.rstrip("/")
    path = "/chat/completions"

    base_per_user = NUM_REQUESTS // NUM_CONCURRENT
    remainder = NUM_REQUESTS - base_per_user * NUM_CONCURRENT

    # Count users that will send at least one request
    active_user_indices = [
        i for i in range(1, NUM_CONCURRENT + 1)
        if base_per_user + (1 if i <= remainder else 0) > 0
    ]
    num_active = len(active_user_indices)

    api_keys: list[str] | None = None
    end_user_ids: list[str] | None = None

    if key_mode == "per_user":
        print(f"Creating {num_active} keys via /key/generate...", flush=True)
        api_keys = await create_keys_for_users(base_url, API_KEY, num_active)
        print(f"Created {len(api_keys)} keys.", flush=True)

    if user_mode == "created":
        run_prefix = f"latency-test-{int(time.time())}-"
        end_user_ids = [
            f"{run_prefix}{i}" for i in active_user_indices
        ]
        print(f"Creating {len(end_user_ids)} end users via /customer/new...", flush=True)
        await create_end_users_for_test(base_url, API_KEY, end_user_ids)
        print(f"Created {len(end_user_ids)} end users.", flush=True)

    print(f"=== {NUM_REQUESTS} requests, {NUM_CONCURRENT} users, fire-as-fast-as-possible ===", flush=True)
    print(f"User mode: {user_mode}, Key mode: {key_mode}", flush=True)
    print("Latency = time from request start until last byte of response received", flush=True)

    _results.clear()
    tasks = []
    key_idx = 0
    end_user_idx = 0
    for user_idx in range(1, NUM_CONCURRENT + 1):
        count = base_per_user + (1 if user_idx <= remainder else 0)
        if count > 0:
            if key_mode == "per_user" and api_keys is not None:
                headers = {
                    "Authorization": f"Bearer {api_keys[key_idx]}",
                    "Content-Type": "application/json",
                }
                key_idx += 1
            else:
                headers = {
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                }
            if user_mode == "none":
                payload = {"model": MODEL, "messages": MESSAGES}
            elif user_mode == "created" and end_user_ids is not None:
                payload = {
                    "model": MODEL,
                    "messages": MESSAGES,
                    "user": end_user_ids[end_user_idx],
                }
                end_user_idx += 1
            elif user_mode == "sequential":
                payload = {"model": MODEL, "messages": MESSAGES, "user": str(user_idx)}
            else:  # random (default)
                payload = {"model": MODEL, "messages": MESSAGES, "user": str(random.randint(1000000000, 9999999999))}
            tasks.append(asyncio.create_task(run_user(user_idx, count, base_url, path, payload, headers)))
    await asyncio.gather(*tasks)

    all_results = _results
    if all_results:
        total = len(all_results)
        success_count = sum(1 for _, ok in all_results if ok)
        failed_count = total - success_count
        all_latencies = [lat for lat, ok in all_results if ok]
        if all_latencies:
            n = len(all_latencies)
            avg = sum(all_latencies) / n
            sorted_lat = sorted(all_latencies)
            p95 = sorted_lat[int((n - 1) * 0.95)]
            p99 = sorted_lat[int((n - 1) * 0.99)]
            print(f"\n[OK] {success_count} succeeded, [FAIL] {failed_count} failed")
            print(f"\nLatency (successful):")
            print(f"  min:  {min(all_latencies):.3f}s")
            print(f"  avg:  {avg:.3f}s")
            print(f"  max:  {max(all_latencies):.3f}s")
            print(f"  p95:  {p95:.3f}s")
            print(f"  p99:  {p99:.3f}s")
            print("\nRequests above threshold (successful only):")
            for threshold in range(1, 11):
                above = sum(1 for t in all_latencies if t > threshold)
                print(f"  Above {threshold}s: {above}/{len(all_latencies)} ({100.0 * above / len(all_latencies):.1f}%)")
        else:
            print(f"\n[OK] {success_count} succeeded, [FAIL] {failed_count} failed")


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(user_mode=args.user_mode, key_mode=args.key_mode))
