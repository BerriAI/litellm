# LIT-7078 validation commands

Product tip: `0690520080fde5f63878ac1b4f0e00cc19843f36`
Merge base: `e4706fa4090ba2f6d88532e63428ff32994a64df`

The canonical CI lint environment was retained in the stopped `litellm-7078-checks` container, with the product worktree mounted at its original path. `run_gates.py` records every lint invocation, and `gate-results.json` plus the individual gate logs record results

```sh
docker exec litellm-7078-checks /opt/lit6634-venv/bin/python /verification/run_gates.py
```

Affected tests, including branch coverage:

```sh
docker exec -w /Users/jvalluru/LiteLLM-work/litellm-lit-7078 litellm-7078-checks /opt/lit6634-tests/bin/python -c 'import pydantic.root_model; import pytest; raise SystemExit(pytest.main(["tests/test_litellm/proxy/_experimental/mcp_server/test_discoverable_endpoints.py", "tests/test_litellm/proxy/_experimental/mcp_server/test_byok_oauth_endpoints.py", "-q", "--disable-warnings", "--cov=litellm.proxy._experimental.mcp_server.discoverable_endpoints", "--cov-branch", "--cov-report=json:/verification/coverage.json", "--cov-report=xml:/verification/coverage.xml"]))'
python3 summarize_coverage.py
```

The first concurrent type-check run exited 247. Running the same gate separately completed successfully:

```sh
docker exec -w /Users/jvalluru/LiteLLM-work/litellm-lit-7078 -e PATH=/opt/lit6634-venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin -e UV_PROJECT_ENVIRONMENT=/opt/lit6634-venv litellm-7078-checks /opt/lit6634-venv/bin/python scripts/type_check_gate.py --base e4706fa409
```

Final source build, from the isolated product worktree:

```sh
docker build --label org.opencontainers.image.revision=0690520080fde5f63878ac1b4f0e00cc19843f36 -t litellm-7078:after .
```

`refresh_live.py` records the Compose startup, exact curl requests, strict-library checks, browser capture, source-hash verification, and static-prefix rerun. `refresh-live.log` records completion. Local credentials remain in the uncommitted `.env` and `auth.curl` files. Both local services and the check container were stopped after validation
