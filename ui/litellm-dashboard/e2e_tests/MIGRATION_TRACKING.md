# UI Manual QA → E2E Migration Tracking

Tracks migration of the UI Manual QA Checklist (`UI Manual QA Checklist 33843b8acdab803cb21bda26c40a274a.md`) to automated Playwright e2e tests under this directory and `tests/proxy_admin_ui_tests/e2e_ui_tests/`.

## Coverage Summary

| Bucket | Items | % of CI-eligible |
| --- | ---: | ---: |
| ✅ Covered (full) | 12 | 30% |
| ◐ Partial (subset / wrong role) | 5 | 12.5% |
| ❌ Not covered | 23 | 57.5% |
| ⏭ CI: Skip (out of scope) | 41 | — |
| 🛠 Env setup (not a test case) | 6 | — |
| **Total checkboxes** | **87** | |

**Overall completion (strict, full coverage only): 12 / 40 = 30%**
**Counting partials as half: 14.5 / 40 = 36%**
**Of all 87 checklist items: 14%**

Legend:
- ✅ — automated e2e test exists and exercises the documented flow
- ◐ — covered functionally but missing role/scenario fidelity (e.g. tested as proxy admin instead of team admin)
- ❌ — no e2e test exists, eligible for migration
- ⏭ — explicitly `CI: Skip` in checklist (real LLM call, real provider creds, load test, etc.)
- 🛠 — env/setup step, not a test case

E2E test files:
- Modern suite: `ui/litellm-dashboard/e2e_tests/tests/`
- Older suite: `tests/proxy_admin_ui_tests/e2e_ui_tests/`

---

## Non root docker test setup (1 item)

| # | Checklist item | Status | Test |
| --- | --- | --- | --- |
| 1 | Build/run non root docker | 🛠 | env setup |

---

## Proxy Admin (26 items — 12 ✅ / 1 ◐ / 1 ❌ / 12 ⏭)

| # | Checklist item | Status | Test |
| --- | --- | --- | --- |
| 1 | Create a team | ✅ | `proxy-admin/teams.spec.ts` → "Create a team" |
| 2 | Invite a user to the team | ✅ | `proxy-admin/teams.spec.ts` → "Invite a user to a team" |
| 3 | Create a key in the team | ✅ | `proxy-admin/keys.spec.ts` → "Create a key in a team" |
| 4 | Regenerate key | ✅ | `proxy-admin/keys.spec.ts` → "Regenerate key" |
| 5 | Update key with new tpm/rpm limits | ✅ | `proxy-admin/keys.spec.ts` → "Update key TPM and RPM limits" |
| 6 | Delete key | ✅ | `proxy-admin/keys.spec.ts` → "Delete key" |
| 7 | Perform a chat in playground with at least 2 different models | ⏭ | CI: Skip |
| 8 | Review logs page (verify playground requests logged) | ⏭ | CI: Skip |
| 9 | Review Usage Page – Top API Keys / chart toggle | ⏭ | CI: Skip |
| 10 | Add New Model with Valid non-reused credentials | ⏭ | CI: Skip |
| 11 | Test Connection with Bad non-reused credentials | ✅ | `modelsPage/addModel.spec.ts` → "Test connection with bad credentials shows failure" |
| 12 | Add specific model → confirm in All Models | ✅ | `modelsPage/addModel.spec.ts` → "Add specific model and verify it appears in All Models" |
| 13 | Add wildcard route | ✅ | `modelsPage/addModel.spec.ts` → "Add wildcard route and verify it appears in All Models" |
| 14 | Add new model w/ reusable credential | ⏭ | CI: Skip |
| 15 | See all models for a specific provider in dropdown | ✅ | `modelsPage/addModel.spec.ts` → "Able to see all models for a specific provider in the model dropdown" |
| 16 | Add OpenAI-compatible model + test request | ⏭ | CI: Skip |
| 17 | Add team-only model → assign to team | ❌ | — |
| 18 | Call model via anthropic pass-through route | ⏭ | CI: Skip |
| 19 | Call bedrock anthropic on test key pane | ⏭ | CI: Skip |
| 20 | Edit team model | ✅ | `modelsPage/addModel.spec.ts` → "Edit team model TPM and RPM limits" |
| 21 | Edit team member for team proxy admin does not belong to | ✅ | `proxy-admin/teams.spec.ts` → "Edit team member for team proxy admin does not belong to" |
| 22 | Delete team | ✅ | `proxy-admin/teams.spec.ts` → "Delete a team" |
| 23 | Create key → test if it works | ⏭ | CI: Skip |
| 24 | See all internal user created keys in team | ✅ | `proxy-admin/keys.spec.ts` → "See internal user keys in team" |
| 25 | Team in org → add member not in any org → edit team models | ◐ | `proxy-admin/teams.spec.ts` → "Team in org - edit team member" — only edits a member, does not assert editing the team's models |
| 26 | Create key → langfuse OTEL → request lands in correct project | ⏭ | CI: Skip |

---

## Router Settings QA (2 items — 0 ✅ / 1 ❌ / 1 ⏭)

| # | Checklist item | Status | Test |
| --- | --- | --- | --- |
| 1 | Add fallback (primary + secondary) → verify it appears | ❌ | — |
| 2 | Test fallback via curl, verify response from fallback model | ⏭ | CI: Skip (real model call) |

---

## MCP (3 items — 0 ✅ / 1 ❌ / 2 ⏭)

| # | Checklist item | Status | Test |
| --- | --- | --- | --- |
| 1 | Add MCP server (Streamable HTTP, no auth, deepwiki URL) | ❌ | — |
| 2 | Open MCP tools | ⏭ | CI: Skip |
| 3 | Try calling a tool | ⏭ | CI: Skip |

---

## Public Model Hub Page Tests (3 items — 0 ✅ / 3 ❌ / 0 ⏭)

| # | Checklist item | Status | Test |
| --- | --- | --- | --- |
| 1 | Making models public → verify on `/ui/model_hub_table` | ❌ | (component-level test exists at `src/components/public_model_hub.test.tsx` but no e2e) |
| 2 | Confirming Tabs Work: Internal (Agent Hub, MCP Hub, Claude Code Plugin Marketplace) | ❌ | — |
| 3 | Confirming Tabs Work: Public (Agent Hub, MCP Hub) | ❌ | — |

---

## Guardrails QA (7 items — section-marked CI: Skip)

All 7 items ⏭. Section header literally says `Guardrails QA: CI: Skip`. Out of scope for the e2e migration.

---

## Logs Page (6 items — section-marked CI: Skip)

All 6 items ⏭. Section header literally says `Logs Page: CI: Skip`. Out of scope.

---

## Other Proxy Admin (4 items — 0 ✅ / 2 ❌ / 2 ⏭)

| # | Checklist item | Status | Test |
| --- | --- | --- | --- |
| 1 | Create a Key with All Proxy Models | ❌ | (existing key spec uses All Team Models) |
| 2 | Create a Key with a specific proxy model | ❌ | — |
| 3 | Use the new key to call proxy model | ⏭ | CI: Skip |
| 4 | Use the new key for `/team/list` curl | ⏭ | CI: Skip |

---

## Team Admin (6 items — 0 ✅ / 3 ◐ / 1 ❌ / 2 ⏭)

None of the existing e2e tests use a Team Admin storage state — every spec authenticates as proxy admin.

| # | Checklist item | Status | Test |
| --- | --- | --- | --- |
| 1 | Able to view all team keys | ◐ | proxy-admin tests cover the page; no Team Admin role test |
| 2 | Able to add member from team | ◐ | covered as proxy admin in `teams.spec.ts`, not as Team Admin |
| 3 | Able to remove member from team | ❌ | — |
| 4 | Able to add / edit / delete model in team | ⏭ | CI: Skip |
| 5 | Able to create team key with all team models | ◐ | covered as proxy admin in `keys.spec.ts`, not as Team Admin |
| 6 | Able to see all team models in test key dropdown | ⏭ | CI: Skip |

---

## Internal User (5 items — 0 ✅ / 4 ❌ / 1 ⏭)

No internal-user-role storage state exists in e2e fixtures.

| # | Checklist item | Status | Test |
| --- | --- | --- | --- |
| 1 | Create Key Flow: internal user, no personal models, not in any team | ❌ | — |
| 2 | Create Key Flow: internal user, no personal models, in a team | ❌ | — |
| 3 | Create key → test if it works | ⏭ | CI: Skip |
| 4 | Able to view team info (no team settings) | ❌ | — |
| 5 | Keys page shouldn't show litellm-dashboard keys | ❌ | — |

---

## Internal User Viewer (3 items — 0 ✅ / 3 ❌)

No internal-user-viewer-role storage state exists in e2e fixtures.

| # | Checklist item | Status | Test |
| --- | --- | --- | --- |
| 1 | Nav shows only allowed items | ❌ | (sidebar.spec.ts only tests proxy admin) |
| 2 | Virtual Keys page hides Create / Regenerate / Reset / Delete | ❌ | — |
| 3 | Teams page hides Members / Team settings | ❌ | — |

---

## Server root path tests (4 items — 0 ✅ / 2 ❌ / 2 🛠)

| # | Checklist item | Status | Test |
| --- | --- | --- | --- |
| 1 | Set `SERVER_ROOT_PATH` in .env | 🛠 | env setup |
| 2 | Restart proxy | 🛠 | env setup |
| 3 | UI loads with `{{host}}/{{SERVER_ROOT_PATH}}/ui/login/` | ❌ | — |
| 4 | Keys page loads (sanity) | ❌ | — |

---

## Security Tests (5 items — 0 ✅ / 1 ◐ / 4 ❌)

| # | Checklist item | Status | Test |
| --- | --- | --- | --- |
| 1 | Invite user → set password → sign in → user info shows new user | ❌ | — |
| 2 | Click logout | ❌ | — |
| 3 | Sign in as Admin → user info shows proxy admin | ◐ | `login.spec.ts` only asserts "Virtual Keys" visible, not user info hover |
| 4 | Sign in as new user → UI loads only that user's info | ❌ | — |
| 5 | `PROXY_LOGOUT_URL=https://www.google.com` → click logout → redirect | ❌ | — |

---

## SSO Tests (2 items — 0 ✅ / 1 ❌ / 1 ⏭)

| # | Checklist item | Status | Test |
| --- | --- | --- | --- |
| 1 | Microsoft SSO sign-in (env: `MICROSOFT_CLIENT_*`) | ⏭ | CI: Skip — needs real SSO |
| 2 | Admin Settings → SCIM → Create SCIM token | ❌ | — |

---

## Claude Code Tests (7 items — all ⏭)

All require live LLM calls / Claude Code CLI integration. Out of scope for UI e2e migration.

---

## Server Tests (3 items — all ⏭)

| # | Checklist item | Status |
| --- | --- | --- |
| 1 | Enable prometheus → make request → check server logs | ⏭ |
| 2 | Remove callbacks before stress testing | ⏭ |
| 3 | Confirm redis op/s vs prior version (locust, 2 instances) | ⏭ load test |

---

## Usage Page Tests (numbered list, not checkboxes)

| # | Item | Status |
| --- | --- | --- |
| 1 | Start Proxy with Large DB → Global Usage returns in ~3s | ⏭ perf, requires large DB |

---

## Recommended next migration targets

Highest-leverage, lowest-friction items to close the gap:

1. **Proxy Admin #17** — team-only model → assign to team. `addModel.spec.ts` already exercises Add Model; just toggle the Team-BYOK switch and pick a team.
2. **Proxy Admin #25 → full** — extend the existing `Team in org - edit team member` test to also click "Edit Models" and save, satisfying the original intent.
3. **Public Model Hub #1–3** — three independent flows, all driven by `/ui` + `/ui/model_hub_table`. Fast wins with stable selectors (component test already exists for behavior parity).
4. **Other Proxy Admin #1–2** — All Proxy Models / specific proxy model. Trivial variants of the existing key creation test.
5. **Router Settings #1** — Add fallback. Single-page form, no role complexity.
6. **MCP #1** — Add MCP server (Streamable HTTP, no auth). Standalone form flow.
7. **Internal User Viewer #1–3** — requires adding a `VIEWER_STORAGE_PATH` fixture; once added, the three nav-restriction asserts are short.
8. **Team Admin #1–3, #5** — requires a Team Admin storage fixture; once added, the existing flows in `teams.spec.ts` and `keys.spec.ts` can be parameterized over roles.
9. **Internal User #1–2, #4–5** — same, requires Internal User storage fixture.
10. **Security Tests #1–5** — covers logout + invite flow + per-user isolation; the invite-user code path was sketched in `tests/proxy_admin_ui_tests/e2e_ui_tests/team_admin.spec.ts` (currently fully commented out) and can be revived.
11. **Server root path #3–4** — needs a separate test job that boots the proxy with `SERVER_ROOT_PATH` set; fold into `globalSetup.ts` matrix.
12. **SSO #2** — SCIM token creation is a pure UI flow and doesn't need real SSO env.

After (1)–(6) the CI-eligible coverage rises from **30% → 60%**. Adding role fixtures (steps 7–9) unlocks another ~12 items and pushes coverage above **85%**.

---

## How to read this when planning

- Treat ⏭ items as a fixed denominator — they will not be migrated; ignore them when computing progress.
- ◐ partials should be promoted to ✅ before chasing new ❌ items, since they often share a fixture or page-object cost.
- Each role-gated section (Team Admin, Internal User, Internal User Viewer) is gated on adding the role's storage state to `globalSetup.ts` + `constants.ts`. Do that once and many items unlock together.

---

## Migration Plan: 28 Eligible Items, Easiest → Hardest

All 5 role storage states are already populated by `globalSetup.ts` and seeded in `seed.sql` (admin, viewer, internal user, internal viewer, team admin). Role-gated items are not blocked on fixture work.

### Tier 1 — extend existing tests, same file (~10–15 min each)

| # | Item | What to add |
| --- | --- | --- |
| 1 | Proxy Admin #25 (◐→✅) | Extend `teams.spec.ts` "Team in org - edit team member" to also click Edit Models + Save |
| 2 | Security #3 (◐→✅) | Extend `login.spec.ts` to assert hover-user-info shows admin email |
| 3 | Other Proxy Admin #1 | New test in `keys.spec.ts` — Create key with "All Proxy Models" (no team) |
| 4 | Other Proxy Admin #2 | New test in `keys.spec.ts` — Create key with one specific model |
| 5 | Proxy Admin #17 | New test in `addModel.spec.ts` — toggle Team-BYOK + select team |

### Tier 2 — new spec, proxy admin role, single-page form (~30 min each)

| # | Item | New file |
| --- | --- | --- |
| 6 | Public Model Hub #2 | `tests/modelHub/internalTabs.spec.ts` |
| 7 | Public Model Hub #3 | `tests/modelHub/publicTabs.spec.ts` |
| 8 | Public Model Hub #1 | `tests/modelHub/makePublic.spec.ts` |
| 9 | MCP #1 | `tests/mcp/addServer.spec.ts` |
| 10 | SSO #2 | `tests/settings/scim.spec.ts` |
| 11 | Router Settings #1 | `tests/settings/fallbacks.spec.ts` |

### Tier 3 — alternative role, storage already populated (~30–60 min each)

| # | Item | Storage |
| --- | --- | --- |
| 12 | Internal User Viewer #1 — nav restrictions | `INTERNAL_VIEWER_STORAGE_PATH` |
| 13 | Internal User Viewer #2 — keys page hides actions | `INTERNAL_VIEWER_STORAGE_PATH` |
| 14 | Internal User Viewer #3 — teams page hides settings | `INTERNAL_VIEWER_STORAGE_PATH` |
| 15 | Internal User #4 — view team info (no settings) | `INTERNAL_USER_STORAGE_PATH` |
| 16 | Internal User #5 — keys page filters dashboard keys | `INTERNAL_USER_STORAGE_PATH` |
| 17 | Internal User #1 — create-key flow, no team | `INTERNAL_USER_STORAGE_PATH` |
| 18 | Internal User #2 — create-key flow, in team | `INTERNAL_USER_STORAGE_PATH` |
| 19 | Team Admin #1 — view all team keys | `TEAM_ADMIN_STORAGE_PATH` |
| 20 | Team Admin #2 — add member | `TEAM_ADMIN_STORAGE_PATH` |
| 21 | Team Admin #3 — remove member | `TEAM_ADMIN_STORAGE_PATH` |
| 22 | Team Admin #5 — create team key | `TEAM_ADMIN_STORAGE_PATH` |

### Tier 4 — multi-context / session flows (~1–2 hr each)

| # | Item | Why harder |
| --- | --- | --- |
| 23 | Security #2 — click logout | Tests session destruction across reload |
| 24 | Security #1 — invite + onboarding link | Needs second browser context for invite link |
| 25 | Security #4 — sign-in as new user | Depends on #24 |

### Tier 5 — env-var / proxy-restart variants (~half-day each)

| # | Item | Needs |
| --- | --- | --- |
| 26 | Server root path #3 — UI loads with SERVER_ROOT_PATH | Separate proxy launch with env var set |
| 27 | Server root path #4 — Keys page loads under root path | Same env variant |
| 28 | Security #5 — `PROXY_LOGOUT_URL` redirect | Separate proxy launch with logout URL set |

---

## Watching tests run locally

`run_e2e.sh` already supports `--headed`; the runner brings up postgres in docker, builds the UI, runs the proxy on host, and the browser launches on host.

```bash
# Stop /dev-up first — run_e2e.sh needs ports 4000 / 5432 / 8090
cd ui/litellm-dashboard
./e2e_tests/run_e2e.sh --headed                       # all tests, visible browser
./e2e_tests/run_e2e.sh --headed -- -g "Create a team" # single test
```

For interactive iteration (time-travel debugging, re-run individual tests, network/console panels):

```bash
npx playwright test --config e2e_tests/playwright.config.ts --ui
```

First run takes ~60–90s (postgres pull + UI build); subsequent iterations are fast.
