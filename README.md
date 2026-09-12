# LIT-7078 verification

Before: e4706fa409 (PR merge base)
After: 0690520080fde5f63878ac1b4f0e00cc19843f36

Both images were built using the repository Dockerfile and ran with isolated Docker Compose PostgreSQL on localhost:47078. The OAuth server named example uses authorization_code, explicit upstream GitHub OAuth URLs, and placeholder client credentials. Discovery does not require a real upstream login. A separate auth_type:none server uses https://knowledge-mcp.global.api.aws for real tool calls.

Read the live metadata with:

```sh
curl -sS http://localhost:47078/.well-known/oauth-authorization-server/mcp/example
curl -sS http://localhost:47078/.well-known/oauth-authorization-server/example/mcp
curl -sS http://localhost:47078/.well-known/oauth-authorization-server/example
curl -sS http://localhost:47078/.well-known/oauth-authorization-server/mcp
```

The first two issuers fail exact matching before and pass after. The compatibility and aggregate documents pass on both commits. The screenshots show the actual standard discovery response in a browser.

For the static-prefix case, set SERVER_ROOT_PATH=/gateway and PROXY_BASE_URL=http://localhost:47078/gateway and recreate the gateway. Request:

```sh
curl -i http://localhost:47078/.well-known/oauth-authorization-server/gateway/example/mcp
```

The legacy document returns 404 before and HTTP 200 with issuer http://localhost:47078/gateway/example/mcp after.

The attached strict-discovery.mjs uses the unmodified oauth4webapi 3.8.8 library. Its npm tarball integrity was verified against validator-package.json. Extract that package into package/ beside the script, copy both into /tmp of the gateway container, then run:

```sh
docker exec litellm-7078-repro-litellm-1 node /tmp/strict-discovery.mjs
docker exec -e REPRO_BASE_URL=http://localhost:47078/gateway litellm-7078-repro-litellm-1 node /tmp/strict-discovery.mjs
```

The script rewrites only the transport origin to localhost:4000 inside the container, retaining the external expected issuer. HTTP is enabled only for this local test.

On both commits, authenticated POST /mcp/aws_knowledge_mcp tools/list returns five tools and tools/call aws_knowledge_mcp-aws___list_regions returns real AWS region data with isError:false. No LLM call is involved. Upstream OAuth login completion is outside this metadata fix.

On the final tip, 484 affected tests pass. Against the merge base, 13 new regression cases fail and 11 compatibility/control cases pass. Local changed executable-line coverage is 2/2; the authorization builder covers 6/6 branches and both route handlers have no branches. These are separate from the repository-wide CI coverage report.

The static-prefix subprocess test uses a temporary UI directory so importing the proxy cannot rewrite checked-out dashboard assets. The isolated worktree remains clean after the affected tests

All 33 required CI checks passed. Greptile scored 5/5, Veria passed, and Cursor Bugbot found no issues, all on `0690520080fde5f63878ac1b4f0e00cc19843f36`. No actionable review threads remain

The completed Codecov report includes all 26 current-tip coverage flags, with 2/2 changed lines covered (100%). Repository-wide coverage is 80.67%, separate from the local touched-builder branch coverage of 6/6
