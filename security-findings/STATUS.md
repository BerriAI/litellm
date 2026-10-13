# Security finding status tracker

Order of work: Critical/High first. Status values: `open`, `fixed-in: #PR`, `accepted-risk: <reason>`, `upstream`(reported upstream).

| id | severity | finding | status |
|----|----------|---------|--------|
| R1-H1 | High | Request-body size limiting is premium-only and off by default; unauthenticated `/login` buffers unbounded bodies | deploy-mitigated: PR14 (production profile enforces proxy read limits; upstream body-limit remains premium-paywalled) |
| R1-H2 | High | Generic passthrough follows redirects, defeating the SSRF validator; `x-pass-*` injects headers past the protection list (IMDS-style chain) | fixed-in: PR7 |
| R1-H3 | High | Any internal user can attach an arbitrary `budget_id` to a key on `/key/generate` and `/key/update` | fixed-in: PR12 |
| R1-H4 | High | `/v1/realtime/calls` bypasses per-key authorization (budgets, rate limits, model access) | fixed-in: PR22 |
| R1-M1 | Medium | Enterprise `allowed_ips` resolves the "client IP" from a raw `X-Forwarded-For` header | open |
| R1-M2 | Medium | Global JWT path disables `aud`/`iss` verification by default (warning-only) | open |
| R1-M3 | Medium | Internal users can self-grant `teams` membership via `/user/update` | open |
| R1-M4 | Medium | Endless-SSE memory accumulation on every stream; no accumulation cap | open |
| R1-M5 | Medium | Pass-through `forward_headers: true` forwards end-user credentials and framing headers upstream; `x-pass-*` injects framing even off | fixed-in: PR7 |
| R1-M6 | Medium | `/v1/images/edits` ignores `max_file_size_mb` and buffers every upload in RAM | fixed-in: PR21 |
| R1-M7 | Medium | `/v1/files` reads the whole upload into memory by default; same for `/v1/audio/transcriptions` | fixed-in: PR21 |
| R1-M8 | Medium | `/prompts/test` renders request-controlled Jinja with no timeout or size bound, from any key | fixed-in: PR20 |
| R1-M9 | Medium | YAML alias-bomb DoS on authenticated YAML endpoints | fixed-in: PR20 |
| R1-M10 | Medium | Client-controlled `metadata.alerting_metadata` rendered into Slack/MS Teams messages unredacted and unbounded | fixed-in: PR19 |
| R1-M11 | Medium | Slack/Teams alert text embeds attacker prose as raw mrkdwn | fixed-in: PR19 |
| R1-M12 | Medium | Plugin session-claim key attacker-computable when `LITELLM_SALT_KEY` unset (empty-string fallback) | fixed-in: PR9 |
| R1-M13 | Medium | Harness docker/local sandboxes: no network isolation, secrets on argv (SDK surface) | open |
| R1-M14 | Medium | Default `docker-compose.yml` publishes Postgres with a hardcoded public password | deploy-config: PR14 ships an env-secret compose profile as the hardened baseline; the upstream default compose is unchanged |
| R1-M15 | Medium | `curl | sh` installer bootstraps uv without integrity verification | open |
| R1-M16 | Medium | Shipped production image runs as root; helm/default compose set no hardening context | deploy-config: PR14 ships a rootless image profile; the upstream defaults still run as root |
| R1-M17 | Medium | `user_url_validation: false` global kill switch turns user-URL fetches unvalidated | open |
| R1-L1 | Low | OAuth2 introspection fails open on missing `active` | fixed-in: PR9 |
| R1-L2 | Low | OAuth2 token and introspection response logged at debug | fixed-in: PR9 |
| R1-L3 | Low | MCP OAuth decode skips `exp` | open |
| R1-L4 | Low | UI session cookie `httponly=False` | open |
| R1-L5 | Low | `/v1/files/{id}/content` relays provider Content-Type/Disposition without attachment/nosniff | open |
| R1-L6 | Low | RAG JSON-ingest b64-decodes before the size cap | open |
| R1-L7 | Low | Unauthenticated open redirects on `/get_image`//`/get_favicon` for admin-set http(s) logo | open |
| R1-L8 | Low | Legacy SHA-256 password verifier never expires | open |
| R1-L9 | Low | `None == None` user_id match in `/key/info` ownership | fixed-in: PR23 |
| R1-L10 | Low | `requester_ip_address` attacker-set from raw XFF | open |
| R1-L11 | Low | A2A `x-a2a-{agent}-{header}` forwards any caller-named header upstream | open |
| R1-L12 | Low | UI-session trust keyed on client-visible `team_id == "litellm-dashboard"` string | open |
| R1-L13 | Low | Policy templates fetched from unversioned `main` at runtime | open |
| R1-L14 | Low | Legacy license verification: import-time remote call, license in URL | open |
| R2-1 | Critical | Cross-tenant raw-id read/delete on files/batches/fine-tune (default config) | fixed-in: PR16 |
| R2-2 | High | Cross-tenant list oracle for files/batches in every config | fixed-in: PR15 |
| R2-3 | High | `/openai/*` passthrough bypasses managed-files; thread/assistant/vector-store ids never guarded | fixed-in: PR17 |
| R2-4 | High | Global cross-tenant exact-match cache keyspace | fixed-in: PR6 |
| R2-5 | High | Caller-supplied `cache_key` gives arbitrary cache read/write (+ spend-counter overwrite if shared Redis) | fixed-in: PR6 |
| R2-6 | High | `/key/regenerate` missing budget/spend/expiry admin gate | fixed-in: PR3 |
| R2-7 | High | `temp_budget_increase`/`temp_budget_expiry` via key metadata bypasses ceiling | fixed-in: PR3 |
| R2-8 | High | `/vertex_ai/live` model ACL skipped when `?model=` absent; setup-frame model unchecked | fixed-in: PR5 |
| R2-9 | High | Team admin repoints deployment `api_base`, harvests static provider credential | fixed-in: PR18 |
| R2-10 | High | Redis outage: rate limits downgrade to per-worker in-memory (fail-open default) | open |
| R2-11 | High | Redis outage: model budgets read spend as 0.0 (fail-open) | open |
| R2-12 | High | Org admin reads all tenants' spend-log rows incl. prompts | fixed-in: PR11 |
| R2-13 | Medium | Login CSRF: fastapi-sso 0.19.0 state not enforced for Google/Microsoft | fixed-in: PR24 (sso_state cookie enforced and re-emitted on Google/Microsoft login) |
| R2-14 | Medium | `/invitation/update` no authz - burn anyone's invite | fixed-in: PR1 |
| R2-15 | Medium | `/key/update` owner shrinks `budget_duration` to seconds | fixed-in: PR3 |
| R2-16 | Medium | `throttle_on_budget_exceeded` metadata injection bypasses typed gate | fixed-in: PR3 |
| R2-17 | Medium | `/user/update` self-update leaves rpm/tpm/models unprotected | fixed-in: PR4 |
| R2-18 | Medium | `/spend/tags` unscoped cross-tenant aggregation to any internal user | fixed-in: PR10 |
| R2-19 | Medium | `/spend/logs/ui` list unscoped for org admins | fixed-in: PR11 |
| R2-20 | Medium | Audit `changed_by` forgeable via self-set key metadata + header | fixed-in: PR10 |
| R2-21 | Medium | Realtime tokens unbound to key, replayable, caller-set TTL, upstream-trusted expiry | fixed-in: PR8 |
| R2-22 | Medium | Realtime WS budget reservation lapses at `request_timeout`; no mid-session enforcement | fixed-in: PR33 (the reservation lease renews for the request task's whole lifetime — sessions outlast request_timeout — and mid-session enforcement re-reads applied counters and cancels an in-flight request the moment it goes over budget) |
| R2-23 | Medium | MCP tool-name shadowing via bare-name mapping | fixed-in: PR25 (bare-name dispatch rejects candidates carrying a different registered server's known prefix; poisoned rows fail closed) |
| R2-24 | Medium | One-click XSS: MCP tool-result `javascript:` href + httponly=false cookie | fixed-in: PR13 |
| R2-25 | Medium | End-user budget bypass by rotating `user` field + unbounded Redis key growth | open |
| R2-26 | Medium | Budget-window GET-then-SET race resets spend counters | open |
| R2-27 | Medium | Dynamic/priority/batch rate limiters fail open on Redis errors | open |
| R2-28 | Medium | PyPI publish = manual twine with personal tokens, no provenance | upstream |
| R2-29 | Medium | Release images built by no auditable pipeline; no signing/attestation | upstream |
| R2-30 | Medium | BerriAI CI: issue-body prompt injection can exfiltrate LITELLM_API_KEY | upstream |
| R2-31 | Medium | 27 stale unverifiable checked-in enterprise wheels drift from source | accepted-risk: not consumed by any runtime path; flagged upstream |
| R2-32 | Medium | Provider-native vector stores unowned, cross-tenant | open |
| R2-33 | Low | Search-tool tester `javascript:` href / window.open | fixed-in: PR13 |
| R2-34 | Low | No script CSP on admin UI; HSTS opt-in | open |
| R2-35 | Low | `/metrics` accepts any virtual key | fixed-in: PR13 |
| R2-36 | Low | Team guardrail submissions accept internal `api_base` (approval fatigue) | open |
| R2-37 | Low | `/routes` unauthenticated route enumeration | fixed-in: PR13 |
| R2-38 | Low | Tracebacks in `/global/spend*` error bodies (needs get_spend_routes grant) | fixed-in: PR13 |
| R2-39 | Low | Invitation expiry date-granularity; non-atomic login-code redemption; tokens in URLs | open |
| R2-40 | Low | CLI SSO user_code in logs; /sso/cli/start LB-IP shared bucket | open |
| R2-41 | Low | WS subprotocol echoes credential into 101 response | fixed-in: PR23 (negotiated subprotocol never reflects key-bearing tokens) |
| R2-42 | Low | vertex_ai/live project/location cost-shifting via query params | open |
| R2-43 | Low | Redis plaintext default + password in debug log | fixed-in: PR23 (debug dump masks password-bearing kwargs; TLS/auth remains deploy-side) |
| R2-44 | Low | `max_parallel_requests` slots released after 1h | closed: documented in-code (slot leak-prune after 1h and the longest-trackable request are annotated at `PARALLEL_REQUEST_SLOT_TTL_SECONDS`; no code change needed; raise the TTL deploy-side if >1h requests must stay slot-reserved) — recorded by PR26 |
| R2-45 | Low | `authorization*` prefix promoted to OAuth2 bearer in MCP admission | fixed-in: PR26 (exact-match promotion only; `authorization-<suffix>` headers no longer shadow Authorization) |
| R2-46 | Low | Duplicate MCP server names: union-grant vs first-match routing | accepted-risk: deployments must register one canonical server per name; tool routing stays first-register-match while server-list grants union duplicates — validated at deploy config, not in code |
| R2-47 | Low | `agent_id` on spend logs client-influenceable | fixed-in: PR26 (attribution resolves from the key and the server-resolved invoked agent; header/body-supplied agent_id is dropped) |
| R2-48 | Low | CODEOWNERS covers only cost map; fork-dependency guard name-only | upstream |

Note: infra rows (R2-28..31, R2-48) are upstream-facing supply-chain findings. We mitigate on our side by pinning digests and verifying cosign when building our image; they are marked `upstream`/`accepted-risk` rather than fork-fixable.

## Disposition of remaining open findings — audit (2026-10-13)

Every row still `open` is assigned one of four buckets. The bucket is the plan; the row stays `open` until the fix or the decision lands.

### Fix next in the fork (code PRs)

- **(Redis-outage and budget-session postures shipped: R2-22 in PR33, R2-25 deploy-config in PR30, R2-26 in PR29, R2-27 in PR27, R2-10/R2-11 in PR27+PR28.)**
- **R2-32** — provider-native vector store ownership: the biggest remaining item (requires ownership checks on every id-bearing native route and the managed-id rewriter for thread/assistant/vector-store ids; list routes + `require_managed_files` OSS support already shipped in PR15).
- (R1-H3 shipped in PR12, R2-9 shipped in PR18.)
- **R1-M1, R1-L10** — one consistent trusted-proxy resolution in `get_request_ip_address`/`requester_ip_address` (the newer services already have the pattern; this is application).
- **R1-M2** — JWT env-var path: default `verify_aud`/`verify_iss` to enabled with an explicit opt-out flag, matching the per-issuer path.
- **R1-M4** — endless-SSE memory accumulation: add a bounded chunk-retention cap. (M6/M7 shipped in PR21; M8/M9 shipped in PR20.)
- (R1-M10/M11 shipped in PR19 — mrkdwn escaping and dropping the attacker-supplied `alerting_metadata` block from non-admin callers.)
- **R2-25, R2-26, R2-27** — budget/rate limiter math/TOCTOU batch (kept in the fix-next list; see also the R2-10/11 sequencing note above).
- **R2-36** — guardrail-submission internal-host validation (peak value; uses `validate_url` allowlist).
- **R2-39, R2-40, R2-42** — remaining open small-lows: invitation expiry granularity / atomic login-code redemption / tokens in URLs; CLI SSO user_code in logs and /sso/cli/start LB-IP bucket; vertex_ai/live query-param cost-shifting.
- **R2-41, R2-43** — shipped in PR23 (subprotocol echo scrub; redis debug-log mask). **R2-44** — no code change; the 1h slot-leak prune is documented in-code, recorded by PR26. **R2-45, R2-47** — shipped in PR26 (exact-match OAuth2 promotion; key-owned agent attribution). **R2-46** — accepted-risk with a single-canonical-name deploy constraint, recorded by PR26.
- **R1-L5, R1-L6, R1-L7, R1-L8, R1-L12, R1-L13** — small fixes batch (nosniff/attachment, RAG decode ordering, get_image redirect allowlist, force legacy-hash migration, None==None in key/info, UI-session marker flag, policy-templates local default). (R1-L9 shipped in PR23.)

### Deploy-config bucket (fix in our deploy PR, not in litellm source)

- **R1-H1** — request-size limiting is enterprise-licensed upstream; our deployment runs premium, so the mitigation is `max_request_size_mb: 100` in the launchdarkly deploy config plus the premium license key. Status stays open until the deploy PR lands, then flips to accepted-risk w/ deploy pointer.
- **R1-M14** — default docker-compose publishes Postgres/`dbpassword9090`. Our deploy does not use that compose; the deploy PR ships a hardened compose (random password, no published ports).
- **R1-M16** — shipped image runs as root. Our image build uses `docker/Dockerfile.non_root` (USER 65534) or a `user:` override in compose.
- **R2-34** — dashboard CSP defaults. Hardening: `default-src 'self'` + frame-ancestors none at the L7 proxy (LB) instead of the app layer; app-side CSP can be destabilizing for Next.js inline scripts.

### Accepted-risk (documented, currently right tradeoff)

- **R1-M13** (harness docker sandbox: secrets on argv + no network lock) — the harness is SDK-only; no proxy route imports it. Network-isolation guidance lands in the deploy docs; accepted until the harness becomes an HTTP surface.
- **R1-M15** (curl|sh uv installer) — we install from `uv.lock` only; report upstream.
- **R1-M17** (`user_url_validation` kill switch) — operator opt-out, default is the secure posture. Guard the default in deploy validation, not codefork-wide.
- **R1-L3** (MCP OAuth decode skips exp) — the embedded key is re-resolved through user_api_key_auth on every request; residual is one legacy-shape path.
- **R1-L4** (UI session cookie httponly=false) — the dashboard reads `document.cookie`; changing this requires an upstream UI rework. Mitigated by PR13's scheme gating (javascript: URLs are now unrenderable from tool results).
- **R1-L11** (A2A x-pass arbitrary headers) — by-design extension with credential + framing header guards; documented.
- **R1-L14** (legacy license remote check) — premium-license flow; our deploy pins to the signature-verified (keypair) license path.

### Upstream-facing

- **R2-28, R2-29, R2-30, R2-31, R2-48** — already marked `upstream`/`accepted-risk`; the fork mitigates by pinning digests, verifying cosign, and never installing from `main`.
