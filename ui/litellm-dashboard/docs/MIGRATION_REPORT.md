# Phase 1 migration report

**Status:** Phase-1 foundation complete + 4 sections migrated. The remaining
29 sections are deferred per the plan's partial-completion allowance (D8)
and will be picked up in subsequent runs of this branch.

**Branch:** `cursor/ui-shadcn-phase1-f9cc564e-1c6d-4f09-bb5c-9b23416c9be2`

**Date of this checkpoint:** 2026-04-23

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
  `no-banned-ui-imports` and `no-raw-tailwind-colors`. Plugin registered
  via a `file:` dep in `package.json`; `npm run lint` replaced
  `next lint` (Next 16 removed the latter).
- **Task 6:** Docs scaffolding (`BLUEPRINT`, `QUIRKS`, `DEVIATIONS`,
  `CYCLES`, `MIGRATION_REPORT`, `blockers/`).
- **Task 7:** New Playwright `parity` project (testDir `./parity`,
  1440×900 viewport, 1% pixel-diff tolerance, auto-starts `npm run dev`
  by default; `PARITY_BASE_URL` env var points at the proxy when needed).
- **Task 8:** Foundation smoke tests all green (build ✓, tsc unchanged,
  lint rules firing as expected, vitest pre-migration 3801/3801).
- **Task 9:** `docs/RECIPE.md` committed (5-layer gating, 7-cycle cap).

### Sections migrated (4 of 33)

All passed the non-Playwright gates (TS, Lint, Vitest, Build).

| # | Section | Leftnav key | Files migrated |
|---|---------|-------------|----------------|
| 1 | Access Groups | `access-groups` | `AccessGroupsPage.tsx`, `AccessGroupsDetailsPage.tsx`, `AccessGroupsModal/{AccessGroupBaseForm,AccessGroupCreateModal,AccessGroupEditModal}.tsx`. Blueprint drafted and locked. |
| 2 | Virtual Keys (stress test) | `api-keys` | `VirtualKeysPage/VirtualKeysTable.tsx`, `key_team_helpers/BudgetWindowsEditor.tsx`. Blueprint stress-test passed — no new patterns required. |
| 3 | Budgets | `budgets` | `budgets/{budget_panel,budget_modal,edit_budget_modal}.tsx`. |
| 6 | Tool Policies | `tool-policies` | `ToolPolicies/PolicySelect.tsx` (single banned-import file in the section). Categorical palette deviation logged. |

### Blueprint status

**🔒 Locked** at end of Section 1 (Access Groups) with a full translation
table, icon map, toast pattern, form pattern (simple + complex RHF/zod
variants), table pattern, and shared layout patterns. Section 2's
stress-test required no additions — the blueprint is immutable for the
rest of phase-1.

### Test results

- `npm run build`: ✓
- `npx tsc --noEmit`: no new errors in source files; pre-existing test-
  fixture type errors unchanged.
- `npm run lint`:
  - `litellm-ui/no-banned-ui-imports` violations: ~900 (all in
    un-migrated sections, expected to clear as those sections are
    migrated).
  - `litellm-ui/no-raw-tailwind-colors` violations: ~3500 (same).
  - No new violations introduced by foundation or migrated sections.
- `npx vitest run` on migrated sections + molecules:
  **170 / 170 tests pass** (AccessGroups, VirtualKeysPage,
  key_team_helpers, budgets, ToolPolicies, molecules).
- Full-tree `npx vitest run`: **3801 / 3801 pass** (confirmed during Task 4).
- Playwright parity specs: committed but **not executed** in this cloud
  sandbox (no proxy / dev-server auth helper). See
  `docs/DEVIATIONS.md` for the explicit skip rationale.

### CLAUDE.md

Repo-root `CLAUDE.md` updated with a new `### UI` section describing the
post-migration stack, lint enforcement, migration-in-progress note, and
phase-2 deferral list. Replaces the old `### UI Component Library`
section.

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

Once all 33 sections are migrated and no file imports any of the above,
Task 45 will `npm uninstall` them in one commit.

## Bundle size delta

- **Baseline** (pre-migration, captured at Task 2 end):
  - Total JS across `out/_next/static`: 15 425 508 bytes.
  - `out/` total: 21 MB.
  - See `docs/baseline-bundle-size.txt` for the full chunk breakdown.
- **Post-migration (partial):** not yet re-measured. The real bundle-size
  win is realized at Task 45 when the deprecated deps are uninstalled.
  The foundation commits alone will add weight (new shadcn primitives,
  new deps) — the net win comes from removing the much larger antd +
  tremor payloads.

## Sections not yet migrated (29 of 33)

4. Tag Management, 5. Vector Stores, 7. Search Tools, 8. Teams,
9. Organizations, 10. Internal Users, 11. MCP Servers, 12. Skills,
13. Guardrails, 14. Models + Endpoints, 15. Projects, 16. Agents,
17. Router Settings, 18. Admin Panel, 19. Cost Tracking, 20. UI Theme,
21. Logging & Alerts, 22. Usage, 23. Logs, 24. Guardrails Monitor,
25. Old Usage, 26. Caching, 27. Playground, 28. Policies,
29. API Playground, 30. Prompts, 31. API Reference, 32. AI Hub,
33. Learning Resources.

These sections still contain imports that the `litellm-ui/no-banned-ui-imports`
rule flags; the rule is stable and will continue to flag them until
they're migrated. The blueprint and recipe are ready for the next run.

## Known phase-2 prerequisites (unchanged)

- **React Query adoption** — `@tanstack/react-query` is installed but
  fetches still use the existing custom hooks / `fetch` patterns. Phase 2
  will do the systematic sweep.
- **File layout** — `src/components/*` stays flat. Phase 2 introduces
  `src/features/` and route-colocated `_lib` / `_components`.
- **Nested detail routes** — e.g. `/ui/key/{keyId}/settings`. Still
  modal-based today.
- **Date libs** — `moment` and `dayjs` are still imported across the
  codebase. Phase 2 consolidates on `date-fns`.
- **ESLint flat config** — phase 1 stays on `.eslintrc.json`. Phase 2
  migrates to `eslint.config.js`.
- **`@tremor/react` charts** — phase 1 scoped them out; phase 2 (or a
  separate chart-migration task) would replace them with `recharts` or
  similar if desired.
- **`DeleteResourceModal`** and other shared chrome components — to be
  migrated in the chrome sweep (Task 43).

## Quirks and deviations

See `docs/QUIRKS.md` and `docs/DEVIATIONS.md` for per-section entries.

**Cross-cutting deviations:**
- Playwright gates 4–5 (parity + snapshots) are committed but not
  executed in this cloud sandbox. Specs are shape-complete and ready for
  reviewer execution.
- `DeleteResourceModal` and other shared chrome components left for the
  final chrome sweep (Task 43).
- `PolicySelect` uses categorical palette (amber/emerald/red) and is
  explicitly exempted from the raw-color rule.
- `AntdGlobalProvider` is a passthrough today; will be deleted in Task 47
  after antd is uninstalled.
- Custom inline `MultiSelect` wrapper introduced in `AccessGroupBaseForm`
  since shadcn has no multi-select primitive. Subsequent sections that
  need one should copy/paste from there until phase 2 introduces a
  dedicated `@/components/ui/multi-select` primitive.

## For the reviewer

- **Start with:** `docs/BLUEPRINT.md` (the translation + pattern library)
  and `docs/RECIPE.md` (the 5-layer gating procedure).
- **Section examples:** `AccessGroupsPage.tsx` /
  `AccessGroupBaseForm.tsx` are the canonical reference for table and
  form patterns respectively. `VirtualKeysTable.tsx` demonstrates the
  pattern at scale.
- **Playwright parity specs:** one committed for Access Groups
  (`e2e_tests/parity/access-groups.spec.ts`). To execute locally, either
  run the proxy (`uv run litellm --config dev_config.yaml --port 4000`)
  and set `PARITY_BASE_URL=http://localhost:4000`, or wire a dev-server
  auth helper and run against `localhost:3000`.
- **To continue:** the blueprint + recipe are stable. Each remaining
  section is an independent unit — run the recipe, commit, push.
