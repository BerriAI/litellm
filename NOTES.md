# Issue #32412 — UI startup fails with `httpx.ReadError` while loading `general_settings` from DB

## 1. Root cause

`ProxyStartupEvent.initialize_scheduled_background_jobs()` resolves `store_model_in_db` by
reading the `general_settings` row straight off the Prisma HTTP query engine:

```python
_db_gs_record = await ConfigRepository(prisma_client).table.find_first(
    where={"param_name": "general_settings"}
)
```

This is the first real DB read the proxy issues after `prisma_client.connect()`, and it is the
only read on the startup path that talks to Prisma without going through
`call_with_db_reconnect_retry`. When the Prisma engine's keep-alive connection has already been
closed (idle-connection reap, engine still settling right after connect; more visible on Windows,
which is what the reporter runs), httpx raises `ReadError` on that first request. The reporter's
own evidence lines up with a dead pooled connection rather than a broken database: the engine's
`/health` endpoint answers 200, and a fresh `db.litellm_config.find_first()` from a separate
process succeeds.

The repo already has a canonical remedy for exactly this failure mode. `httpx.ReadError` is listed
in `DB_CONNECTION_ERROR_TYPES` (`litellm/proxy/_types.py`), `is_database_transport_error()`
classifies it as a transport blip, and `call_with_db_reconnect_retry()` reconnects once and retries
the read. `PrismaClient.get_generic_data`, `_init_sso_settings_in_db`,
`_init_hashicorp_vault_config_override`, and the cache-settings endpoints all use it; this startup
read was left out.

Why startup dies rather than logging and moving on: the `except Exception` around the read swallows
the `ReadError`, but it does nothing to heal the connection. `store_model_in_db` stays `False`, so
control falls into `await proxy_config.init_mcp_servers_from_db()` (and, when the flag is set,
`add_deployment` / `get_credentials`), which hit the same poisoned connection with no guard at all.
That second `ReadError` propagates out of the startup event, and uvicorn prints
"Application startup failed. Exiting."

There is a second, quieter consequence of the same gap: a proxy that persists
`store_model_in_db: true` in the DB silently ignores it whenever the first read blips, so no models
load and the UI comes up empty.

## 2. The fix

Route the startup `general_settings` read through `call_with_db_reconnect_retry`, the same wrapper
every other DB read path uses. On `httpx.ReadError` the helper calls
`prisma_client.attempt_db_reconnect()`, which recreates the Prisma client (killing the stale engine
and its connection pool), then retries the read exactly once.

That single change fixes both symptoms. The retried read returns the row, so `store_model_in_db` is
honored; and because `attempt_db_reconnect` replaced the transport, the downstream startup reads
(`init_mcp_servers_from_db`, `add_deployment`, `get_credentials`) run against a healthy connection
instead of inheriting the poisoned one. The reconnect is singleflight and at-most-one-retry by
construction, so there is no new retry loop or startup stall.

I deliberately left the surrounding `try/except Exception` alone: a genuinely unreachable database
should still not be turned into a hard startup failure by this particular flag lookup.

## 3. Files changed

- `litellm/proxy/proxy_server.py` — wrap the startup `store_model_in_db` lookup in
  `call_with_db_reconnect_retry` (the symbol was already imported at line 327).
- `tests/test_litellm/proxy/test_proxy_server.py` — add
  `test_initialize_scheduled_jobs_recovers_from_transient_read_error`, a regression test that makes
  the first `find_first` raise `httpx.ReadError`, then asserts one reconnect happened, the read was
  retried, and the DB-persisted `store_model_in_db=True` took effect (`add_deployment` awaited,
  `init_mcp_servers_from_db` not called).

## 4. Risk / uncertainty

Low risk. The change is one call site, adds no new behavior on the success path (`coro_factory()`
is awaited exactly as before and returns the same value), and reuses a helper already exercised by
`tests/test_litellm/proxy/db/test_exception_handler_reconnect_retry.py`. On a non-transport error
the helper re-raises immediately, so nothing that used to fail fast now retries. If the client has
no `attempt_db_reconnect` attribute (partial stand-ins in tests), the helper re-raises rather than
blowing up.

The main uncertainty is that I could not reproduce the reporter's Windows environment, so the
attribution of the `ReadError` to a stale engine connection rests on their evidence (engine health
200, identical query succeeding out-of-band) plus the fact that this exception type is already
treated as a recoverable transport blip everywhere else in the codebase. If the underlying cause on
their machine were instead a hard, persistent transport failure, the reconnect would fail, the
`except` would swallow it as before, and startup would still die on the next unguarded read; the
logs would then show the `DB transport error on read; attempting reconnect-and-retry` warning
followed by a failed reconnect, which is strictly more diagnostic than today's silent debug line.

I did not touch the `isinstance(_db_gs_record.param_value, dict)` check, which skips the override
whenever Prisma hands back `param_value` as a JSON string (`ConfigRepository.get_param` handles
that case; this call site does not). That looks like a real, separate bug, but it is out of scope
for this issue.

## 5. Verification

```
$ uv run python -m pytest tests/test_litellm/proxy/test_proxy_server.py \
      tests/test_litellm/proxy/db/test_exception_handler_reconnect_retry.py -q
234 passed, 15 warnings in 9.01s
```

To confirm the test actually pins the bug rather than the implementation, I stashed the source
change and re-ran just the new test against the old code:

```
$ git stash push litellm/proxy/proxy_server.py
$ uv run python -m pytest tests/test_litellm/proxy/test_proxy_server.py \
      -k recovers_from_transient_read_error -q
FAILED tests/test_litellm/proxy/test_proxy_server.py::test_initialize_scheduled_jobs_recovers_from_transient_read_error
1 failed, 223 deselected
```

It fails on the unpatched code (no reconnect, no retry, `store_model_in_db` never set) and passes
with the fix. The test drives the real `call_with_db_reconnect_retry` and a real `httpx.ReadError`
rather than mocking the helper, so it also covers the classification of `ReadError` as a
reconnect-worthy transport error.
