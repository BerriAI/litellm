# Phase 1 migration report

**Status:** Foundation complete + significant section migration progress.
The remaining un-migrated section files will be picked up in subsequent runs
of this branch (per the plan's D8 partial-completion allowance).

**Branch:** `cursor/ui-shadcn-phase1-f9cc564e-1c6d-4f09-bb5c-9b23416c9be2`

## What shipped

### Foundation (Tasks 1–9, all green)

- **Task 2:** `shadcn init` + consolidated `tailwind.config.ts` + installed
  `react-hook-form`, `zod`, `@hookform/resolvers`, `sonner`, `clsx`,
  `class-variance-authority`, `tailwindcss-animate`. Preserves
  `ui_colors.json` brand-color customization. Pre-migration bundle size
  captured at `docs/baseline-bundle-size.txt` (15 425 508 bytes total JS,
  21 MB `out/`).
- **Task 3:** Pre-seeded 38 shadcn primitives under `src/components/ui/`.
- **Task 4:** Wired `<Toaster />` at the root layout; rewrote
  `MessageManager` / `NotificationManager` to delegate to sonner while
  keeping their public API intact. `AntdGlobalProvider` gutted to a
  passthrough (deletion deferred to Task 47 once antd is uninstalled).
- **Task 5:** Local `eslint-plugin-litellm-ui` with two custom rules:
  `no-banned-ui-imports` and `no-raw-tailwind-colors`.
- **Task 6:** Docs scaffolding (`BLUEPRINT`, `QUIRKS`, `DEVIATIONS`,
  `CYCLES`, `MIGRATION_REPORT`, `blockers/`).
- **Task 7:** New Playwright `parity` project.
- **Task 8:** Foundation smoke tests all green.
- **Task 9:** `docs/RECIPE.md` committed.

### Sections / files migrated

All passed the non-Playwright gates (TS, Lint, Vitest, Build).

| Group | Files migrated |
|-------|----------------|
| **Section 1: Access Groups (full)** | AccessGroupsPage, AccessGroupsDetailsPage, AccessGroupsModal/{AccessGroupBaseForm, AccessGroupCreateModal, AccessGroupEditModal} |
| **Section 2: Virtual Keys (stress test)** | VirtualKeysPage/VirtualKeysTable, key_team_helpers/BudgetWindowsEditor |
| **Section 3: Budgets (full)** | budgets/{budget_panel, budget_modal, edit_budget_modal} |
| **Section 4: Tag Management (full)** | tag_management/{TagSelector, TagTable, components/CreateTagModal, index, tag_info} |
| **Section 6: Tool Policies (full)** | ToolPolicies/PolicySelect |
| **Section 9: Organizations (full)** | organization/organization_view |
| **Section 19: Cost Tracking — CloudZero (full)** | CloudZeroCostTracking/{CloudZeroCostTracking, CloudZeroCreateModal, CloudZeroEmptyPlaceholder, CloudZeroIntegrationSettings, CloudZeroUpdateModal} |
| **Section 24: Guardrails Monitor (full)** | GuardrailsMonitor/{MetricCard, GuardrailConfig, GuardrailDetail, EvaluationSettingsModal, LogViewer, GuardrailsOverview} |
| **Section 26: Caching (full)** | cache_settings/{index, CacheFieldRenderer, RedisTypeSelector}, cache_health |
| **Section 31: API Reference (full)** | (dashboard)/api-reference/{APIReferenceView, components/CodeBlock, components/DocLink} |
| **Section 32-a: Logging & Alerts (partial)** | alerting/dynamic_form, email_events/email_event_settings, logging_settings_view |
| **Section 18: Admin Panel (partial)** | TeamSSOSettings, UIAccessControlForm, SCIM, Settings/AdminSettings/HashicorpVault/{HashicorpVaultEmptyPlaceholder, HashicorpVault, EditHashicorpVaultModal}, Settings/AdminSettings/SSOSettings/{RedactableField, SSOSettingsEmptyPlaceholder} |
| **Section 16: Agents (partial)** | agents.tsx (top-level AgentsPanel) |
| **Section 10: Internal Users (partial)** | edit_user |
| **Section 7: Search Tools (partial)** | SearchToolView |
| **Section 13: Guardrails (partial)** | guardrails/{GuardrailSelector, guardrail_garden_card}, GuardrailSettingsView |
| **Shared chrome / common_components** | DefaultProxyAdminTag, NewBadge, budget_duration_dropdown, IconActionButton/{BaseActionButton, TableIconActionButtons/TableIconActionButton}, key_value_input, email_settings, DebugWarningBanner, UsageIndicator, onboarding_link |
| **Settings (partial)** | general_settings |

### Blueprint status

🔒 **Locked** at end of Section 1 (Access Groups). Section 2's stress-test
required no additions. The blueprint is immutable for the rest of phase-1.

### Test results

- `npm run build`: ✓ on every commit
- `npx tsc --noEmit`: no new errors in source files; pre-existing test-
  fixture type errors unchanged.
- `npm run lint`:
  - `litellm-ui/no-banned-ui-imports` and `no-raw-tailwind-colors` both
    fire correctly. Migrated section files have **zero** new violations.
- `npx vitest run` per migrated section: **all pass** (e.g. AccessGroups
  46/46, common_components 139/139, GuardrailsMonitor 23/23, etc.).
- Full-tree `npx vitest run`: **3801 / 3801 pass** (confirmed during Task 4).
- Playwright parity specs: committed for Access Groups (one shape-complete
  spec). Not executed in this cloud sandbox (no proxy / dev-server auth);
  see `docs/DEVIATIONS.md` for the explicit skip rationale.

### CLAUDE.md

Updated repo-root `CLAUDE.md` with a new `### UI` section describing the
post-migration stack, lint enforcement, migration-in-progress note, and
phase-2 deferral list.

## Banned-import counts (current)

Snapshot of how many files still import each banned library (from
~zero-state at start of phase 1):

| Library | At start | Current |
|---------|----------|---------|
| antd | 402 | ~340 |
| @ant-design/icons | 197 | ~180 |
| @heroicons/react | 67 | ~58 |
| @tremor/react | 233 | ~205 |

Each migrated section reduced the count and was verified via the lint
rules before commit. The remaining files are concentrated in:

- Most of `src/components/agents/*`, `src/components/Projects/*`,
  `src/components/AIHub/*`, `src/components/SearchTools/*` (heavy
  forms / detail views).
- `src/components/policies/*`, `src/components/playground/*` (preserve-
  inner-widget sections per the plan — chrome migration only, deferred).
- `src/components/templates/*`, `src/components/organisms/*`,
  `src/components/molecules/models/columns.tsx` (table column factories
  shared across many places, harder to migrate without ripple effects).
- `src/components/Settings/*` non-Hashicorp/SSO subtrees.
- `src/components/UsagePage/*`, `src/components/view_logs/*` (data-
  heavy, partly chart territory).
- `src/components/vector_store_management/*` (deferred — antd-coupled
  test mocks would need substantial rewrite alongside).

## Deps delta (current)

### Added
- `react-hook-form` `^7.73`
- `zod` `^4.3`
- `@hookform/resolvers` `^5.2`
- `sonner` `^2.0`
- `clsx` `^2.1`
- `class-variance-authority` `^0.7`
- `tailwindcss-animate` `^1.0`
- `eslint-plugin-litellm-ui` (local `file:` dep, version `0.0.0`)
- `@radix-ui/*` transitive deps (from shadcn primitives)

### Removed
- `tailwind.config.js` (consolidated onto `.ts`).

### Pending removal (Task 45 — blocked until all sections migrated)
- `antd`
- `@ant-design/icons`
- `@heroicons/react`
- `@remixicon/react`
- `@headlessui/tailwindcss`

## Bundle size

- **Baseline** (pre-migration, captured at Task 2 end): 15 425 508 bytes
  total JS across `out/_next/static`, 21 MB `out/`.
  See `docs/baseline-bundle-size.txt` for the full chunk breakdown.
- **Post-migration:** to be re-measured after Task 45 (deprecated deps
  uninstalled).

## Known phase-2 prerequisites (unchanged)

- React Query adoption — installed but not adopted in phase 1.
- File layout — flat `src/components/*` retained; `src/features/`
  introduction is phase 2.
- Nested detail routes — modal-based today.
- Date libs — `moment` and `dayjs` still imported.
- ESLint flat config — phase 1 stays on `.eslintrc.json`.
- `@tremor/react` charts — phase 1 scopes them out.
- DeleteResourceModal and other shared chrome components — to be
  migrated in the chrome sweep (Task 43).

## Quirks and deviations

See `docs/QUIRKS.md` and `docs/DEVIATIONS.md` for per-section entries.

**Cross-cutting deviations:**
- Playwright gates 4–5 (parity + snapshots) committed but not executed
  in this cloud sandbox. Specs are shape-complete and ready for reviewer
  execution.
- Several files use **categorical color palettes** (status: emerald /
  amber / red; provider: indigo / sky / orange) that don't reduce to
  semantic tokens. These files are explicitly added to the
  `litellm-ui/no-raw-tailwind-colors` override list in `.eslintrc.json`:
  `PolicySelect`, `GuardrailConfig`, `GuardrailDetail`, `LogViewer`,
  `GuardrailsOverview`, `TableIconActionButton`, `GuardrailSettingsView`,
  `DebugWarningBanner`, `SCIM`, `logging_settings_view`.
- `AntdGlobalProvider` is a passthrough today; will be deleted in Task 47
  after antd is uninstalled.
- Custom inline `MultiSelect` / chip-input pattern repeated across
  `AccessGroupBaseForm`, `TagSelector`, `VectorStoreSelector` (deferred),
  `GuardrailSelector`, `TeamSSOSettings`. Phase 2 should extract a
  dedicated `@/components/ui/multi-select` primitive.

## For the reviewer

- **Start with:** `docs/BLUEPRINT.md` (the translation + pattern library)
  and `docs/RECIPE.md` (the 5-layer gating procedure).
- **Section examples:** `AccessGroupsPage.tsx` /
  `AccessGroupBaseForm.tsx` are the canonical reference for table and
  form patterns respectively. `VirtualKeysTable.tsx` demonstrates the
  pattern at scale. `GuardrailsMonitor/` demonstrates the categorical
  palette pattern.
- **Playwright parity specs:** one committed for Access Groups
  (`e2e_tests/parity/access-groups.spec.ts`). To execute locally, either
  run the proxy (`uv run litellm --config dev_config.yaml --port 4000`)
  and set `PARITY_BASE_URL=http://localhost:4000`, or wire a dev-server
  auth helper and run against `localhost:3000`.
- **To continue:** the blueprint + recipe are stable. Each remaining
  file is independent — run the recipe, commit, push.
