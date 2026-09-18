# Isolated MCP SDK2 dependency gate

This is a development environment for the SDK2 migration. Production MCP/proxy extras and `uv.lock` continue to select SDK1. Installing this candidate does not establish public `MCPClient` or gateway compatibility with SDK2

Build the root wheel and its workspace companions from one checkout:

```bash
uv build --wheel --out-dir /tmp/mcp-wheels
uv build --wheel --package litellm-enterprise --out-dir /tmp/mcp-wheels
uv build --wheel --package litellm-proxy-extras --out-dir /tmp/mcp-wheels
```

Use the root wheel's exact filename in this command. The environment path must not already exist:

```bash
uv run --isolated --no-project --python 3.12 tests/mcp_dependency_tests/runner.py check \
  --wheel /tmp/mcp-wheels/litellm-1.103.0-cp310-abi3-linux_x86_64.whl \
  --profile mcp --mode locked --python 3.12 --environment /tmp/mcp2-dev
```

Profiles are `core`, `mcp` and `proxy`; modes are `minimum` and `locked`. CI installs all six combinations on Python 3.10–3.14. Exact interpreter patch versions are recorded in `candidate.toml` and provisioned through the pinned CI uv tool's managed-Python downloads

Run adapter development commands with the candidate environment's interpreter. Running `uv run` against the root project selects the ordinary SDK1 environment instead. The gate intentionally does not start a gateway or call a remote tool

## What the gate proves

The runner derives dependencies, extras and supported Python versions from wheel metadata, carrying forward the root security constraints and overrides. The candidate adds HTTPX2 and Pydantic floors and overrides only the MCP version. Proxy checks include same-checkout enterprise/proxy-extras wheels, matching the repository workspace rather than omitting packages unavailable on the public index

Snapshot installation enforces archive hashes. The current local wheels are installed without dependency resolution afterward, and the complete installed-version inventory must match the snapshot and those wheels. The deliberate MCP override means this is not a clean public-extra installation claim. SDK2 public-client imports and gateway behavior remain a mandatory later integration gate

Checks require imports from the isolated wheel, distinct HTTPX/HTTPX2 client types, valid and invalid MCP model handling, alias-preserving serialization, and package footprint reports. Core checks additionally execute the existing no-extra smoke runner and reject MCP/HTTPX2 packages. Its base-only guard must never run against an MCP/proxy environment

HTTPX remains owned by existing LiteLLM consumers. HTTPX2 is owned by the candidate MCP SDK integration; removing HTTPX globally is not part of this migration. LangChain MCP adapters 0.2.1 remain in the SDK1 test environment: their requirements resolve with SDK2, but their `RequestContext` import fails. Version 0.3.2 excludes MCP2. These observations cover those two versions only

CI measures runner coverage during actual installs. It measures isolated wheel checks in copies of already verified environments with coverage instrumentation added; original inventory reports stay unchanged

## Updating snapshots

Use CI's uv version (0.10.9). Set an absolute cutoff in `candidate.toml` consistent with the root dependency-age policy, review advisories, then run `lock` for each profile/mode with the newly built wheel:

```bash
uv run --isolated --no-project --python 3.12 tests/mcp_dependency_tests/runner.py lock \
  --wheel /tmp/mcp-wheels/litellm-1.103.0-cp310-abi3-linux_x86_64.whl \
  --profile mcp --mode locked
```

The fingerprint rejects snapshots from different root/companion wheel requirements, security policies or the cutoff. Updating wheel version alone does not require relocking; changing its dependency metadata does. Inspect the lock diff and rerun all actual installations after refresh. Minimum versions characterize the declared support boundary; they are not a recommendation to deploy old package versions or evidence of security clearance

## Integration and retirement

LIT-7738 owns HTTP/auth and connection lifetime, LIT-7739 signing, and LIT-7740 public imports, constructors, callbacks, HTTP/SSE/stdio parity and clean SDK2 packaging without overrides. Preserve shared credential/fault policy and the secured SDK1 release while the SDK2 candidate is tested. Modern advertisement stays disabled

Implementation tickets own matching legacy/security tests and image/config rollback evidence. LIT-7754 coordinates cohort size, observation, error/latency thresholds, session affinity and draining, and compatibility of database/cache/token state written during the canary. Never shadow side-effecting tool calls. Changing the production default and retiring SDK1 are separate gates; legacy protocol retirement retains its announced support window and traffic-observation requirement

Remove candidate overrides only when normal SDK2 wheel/image packaging replaces them. Keep useful compatibility checks. No failed or missing runtime case is a dependency-gate pass, and an additive gate alone does not satisfy the original LIT-7737 requirement to activate SDK2 in public extras
