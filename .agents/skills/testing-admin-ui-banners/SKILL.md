---
name: testing-admin-ui-banners
description: How to run the LiteLLM Admin UI dev server against a live proxy (including a second BEFORE/base worktree) to test dashboard shell banners and /health/readiness/details driven UI state.
---

# Testing Admin UI dashboard banners against a live proxy

## Bring up AFTER (branch under test)

```
sudo service postgresql start
cd <repo> && (setsid uv run --no-sync litellm --config litellm/proxy/dev_config.yaml --detailed_debug --port 4000 > /tmp/proxy.log 2>&1 < /dev/null &)
cd ui/litellm-dashboard && (npm run dev > /tmp/ui_dev.log 2>&1 &)   # port 3000
```

Proxy startup takes ~45-60s before `/health/readiness/details` answers. Log in at
http://localhost:3000/ (it redirects to the proxy's login page) with `admin` /
the `general_settings.master_key` from `litellm/proxy/dev_config.yaml` (`sk-1234` by default).
In dev (`NODE_ENV=development`) the UI defaults its API base to `http://localhost:4000`, so no
extra env var is needed for the main dev server.

Launcher gotcha: if an `exec` shell call runs longer than ~10s it gets backgrounded and can take
the freshly spawned proxy with it. Keep the launch command short (`setsid ... & ; sleep 6`) and
poll readiness in a separate call.

## Bring up BEFORE (base commit) side by side

```
git worktree add /home/ubuntu/repos/litellm-base <base-branch>
cp -al <repo>/ui/litellm-dashboard/node_modules /home/ubuntu/repos/litellm-base/ui/litellm-dashboard/node_modules
```

Do NOT symlink `node_modules` into a worktree: Turbopack panics with
"Symlink [project]/node_modules is invalid, it points out of the filesystem root". A hardlink copy
(`cp -al`) works and is fast.

Run the base proxy with the main venv but the base source tree, and point the base UI at it:

```
cd /home/ubuntu/repos/litellm-base && PYTHONPATH=$PWD <repo>/.venv/bin/python -m litellm.proxy.proxy_cli --config <repo>/litellm/proxy/dev_config.yaml --detailed_debug --port 4001
cd /home/ubuntu/repos/litellm-base/ui/litellm-dashboard && NEXT_PUBLIC_BASE_URL=http://localhost:4001 npm run dev -- --port 3001
```

`PYTHONPATH` wins over the editable install, so the base proxy really runs base code (verify with
`python -c "import litellm; print(litellm.__file__)"`).

## Banner-specific notes

Dashboard shell banners (`DebugWarningBanner`, `NoRedisWarningBanner`, `LicenseExpiryBanner`) all
read `useHealthReadinessDetails`, which has `staleTime: 5 min` and `retry: false`. After restarting
the proxy with different env, hard-reload the page (ctrl+shift+r) or the cached readiness payload
keeps the old banner state. Running the proxy with `--detailed_debug` always shows the yellow debug
banner, which is a handy control: if it is present but the banner under test is not, the readiness
call succeeded and the banner condition really is false.

Coordination Redis (`litellm.proxy.proxy_server.redis_usage_cache`, which drives
`show_no_redis_warning`) is NOT populated by `REDIS_HOST`/`REDIS_PORT` alone: the env fallback only
runs inside `_init_cache`, which requires a cache block in the config. To get a real coordination
Redis, run `docker run -d -p 6379:6379 redis:7` and add to the config:

```
litellm_settings:
  cache: true
  cache_params:
    type: redis
    host: localhost
    port: 6379
```

`general_settings.coordination_redis` is the other supported path.

## Devin Secrets Needed

None for banner/UI-state testing; the proxy boots with the bundled dev config and a local Postgres.
Provider keys are only needed when a test actually issues LLM requests.
