# PR2 0f89cd086b5 review findings, deterministic repro (verification only, no source edits)

Head 0f89cd086b5a77a59c7631292b228959974f5986, parent f594c55059425e024a12200e8c3d0cbf00814b8e.
Script `greptile_repro.py` lives outside the repository. It drives the shipped `PromptCacheAlertLogger` and the shipped
`STREAK_SCRIPT` on a real Redis. Each callback is awaited to completion before the next is delivered, so arrival
order is explicit. No sleeps, no provider calls (case 3 answers api.anthropic.com with respx).

    redis-server --port 6392 --save "" --appendonly no --daemonize yes
    cd <worktree at 0f89cd086b5>
    env -u ANTHROPIC_API_KEY PYTHONPATH=$PWD REDIS_HOST=127.0.0.1 REDIS_PORT=6392 OUT=greptile_repro.json \
      python greptile_repro.py

Request times (seconds after start): proof P 0, miss A 10, hit H 20, miss B 30, miss C 40. `X@wN:k` means the
callback for X ran on logger N and k incidents were saved in total after it.

| Case | Callback arrival | Incidents saved | Result |
|---|---|---|---|
| 1 control, chronological | P A H B C | 0 | correct |
| 1 finding, hit arrives last | P@w0:0 A@w0:0 B@w0:0 C@w0:1 H@w0:1 | 1, saved at C before H arrives | REPRODUCED, false incident. A, B, C are not consecutive, H read the cache between A and B |
| 2 control, H reaches Redis on w0, C on w1 | P A B H C | 0 | correct |
| 2 control, H fails Redis on w0, C also on w0 | P A B H C | 0 | correct, w0 replays its lost reset first |
| 2 finding, H fails Redis on w0, C on w1 | P@w0:0 A@w0:0 B@w1:0 H@w0:0 C@w1:1 | 1 | REPRODUCED, false incident. The lost reset exists only in w0 memory, w1 counts the third miss on the old shared streak |
| 3 plain healthy hit through `litellm.acompletion` | n/a | n/a | detector script called once with `hit` |
| 3 same request with `no-log: true` | n/a | n/a | REPRODUCED, detector script called 0 times after logging drained, so no hit or reset reaches the streak |

Case 3 cause: `Logging.should_run_callback` skips every callback on a `no-log` request unless the class name contains
`_PROXY_`. `PromptCacheAlertLogger` does not. Only the success path was run here. The failure path goes through the
same predicate, which is read from the source and not executed.

Limits: the saved incident in cases 1 and 2 is the `AlertStore.record` call on an in-memory store, not a Postgres
row. Redis and the Lua script are real. Case 2 injects one `redis.exceptions.ConnectionError` on w0's hit call.

## Replay correction for PR2_REPLAY_0f89cd0.md

`gen_cfg.py` takes `name alerts redis alt`. Regenerating with these calls gives files byte-identical to the ones used:

    python gen_cfg.py ON 1 1 0
    python gen_cfg.py ON2 1 1 1
    python gen_cfg.py OFF 0 1 0
    python gen_cfg.py NOREDIS 1 0 0

## Completion evidence for the live negative cases (fixture upstream, no paid calls)

`drv.py` waits 0.5 s and 3 s for callbacks. Those waits alone do not prove a negative outcome. The four break cases
and two controls were rerun on the head proxy (two workers, Redis 6392) and the detector's Redis keys read after each,
in `e2e/head_negatives_state.jsonl`:

| Case | Rows | Redis after the run | Reading |
|---|---|---|---|
| HTTP 500 between misses | 0 | `reset` key set, 2 misses in the set | the failure callback was processed and reset the deployment |
| missing usage between misses | 0 | `reset` key set, 2 misses | processed |
| streamed request between misses | 0 | `reset` key set, 2 misses | processed |
| hit between misses | 0 | no `reset` key, epoch moved past the proof, 2 misses | hit processed |
| never proven, 5 misses | 0 | no state key, 0 misses | nothing tracked |
| control, proof and 2 misses | 0 | state key, 2 misses | below threshold |

Observational only, with no completion signal of their own, backed by unit and real-Redis tests: unsupported surfaces,
providers, two breakpoints and 1h TTL (case 10), default off (20), no Redis (21), config and account change (14, 15),
`FLUSHALL` (16). Cases 17 and 19 have log warnings as completion signals. Not run live: None-client reload gap, proof
expiry, quiet-period expiry.
