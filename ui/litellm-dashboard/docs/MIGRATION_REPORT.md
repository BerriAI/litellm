# Phase 1 migration report

**Status:** In progress. Foundation complete; sections migrated incrementally.
**Branch:** `cursor/ui-shadcn-phase1-f9cc564e-1c6d-4f09-bb5c-9b23416c9be2`

## Foundation (Tasks 1–9)

All nine foundation tasks committed:

- **Task 2:** `shadcn init` + consolidated `tailwind.config.ts` + installed
  `react-hook-form`, `zod`, `@hookform/resolvers`, `sonner`, `clsx`,
  `class-variance-authority`, `tailwindcss-animate`. Preserves
  `ui_colors.json` brand-color customization. Baseline bundle size
  captured at `docs/baseline-bundle-size.txt` (~15.4 MB total JS).
- **Task 3:** Pre-seeded 38 shadcn primitives under `src/components/ui/`.
- **Task 4:** Wired `<Toaster />` at the root layout; rewrote
  `MessageManager` / `NotificationManager` to delegate to sonner while
  keeping their public API intact (every call site still works).
  `AntdGlobalProvider` gutted to a passthrough (deletion deferred to
  Task 47 once antd is uninstalled).
- **Task 5:** Local `eslint-plugin-litellm-ui` with two custom rules:
  `no-banned-ui-imports` (antd, @ant-design/icons, @heroicons, @remixicon,
  non-chart @tremor) and `no-raw-tailwind-colors` (semantic tokens only).
  Plugin registered in `.eslintrc.json`; `npm run lint` now uses direct
  eslint (replaced the broken `next lint` invocation).
- **Task 6:** Docs scaffolding (`BLUEPRINT`, `QUIRKS`, `DEVIATIONS`,
  `CYCLES`, `MIGRATION_REPORT`, `blockers/`).
- **Task 7:** New Playwright `parity` project (testDir `./parity`, 1440×900
  viewport, 1% pixel-diff tolerance). Default targets `localhost:3000`
  via auto-started `npm run dev`; `PARITY_BASE_URL` env var points at
  the proxy when needed.
- **Task 8:** Foundation smoke tests green (build ✓, tsc ✓, lint rules
  firing as expected, vitest 3801/3801 pre-migration).
- **Task 9:** `docs/RECIPE.md` committed.

## Sections (Tasks 10–42, 33 total)

Status at last checkpoint:

### Done (all 5 gates layers 1–3 green; layers 4–5 skipped per DEVIATIONS.md)

1. **Access Groups** (`access-groups`) — 5 files, blueprint draft + lock.
2. **Virtual Keys** (`api-keys`) — 2 directly-owned files. Blueprint
   stress-test passed; no new patterns needed.
3. **Budgets** (`budgets`) — 3 files.
6. **Tool Policies** (`tool-policies`) — 1 file (`PolicySelect.tsx`) with
   documented categorical-palette deviation.

### Pending

4. Tag Management, 5. Vector Stores, 7. Search Tools, 8. Teams,
9. Organizations, 10. Internal Users, 11. MCP Servers, 12. Skills,
13. Guardrails, 14. Models + Endpoints, 15. Projects, 16. Agents,
17. Router Settings, 18. Admin Panel, 19. Cost Tracking, 20. UI Theme,
21. Logging & Alerts, 22. Usage, 23. Logs, 24. Guardrails Monitor,
25. Old Usage, 26. Caching, 27. Playground, 28. Policies,
29. API Playground, 30. Prompts, 31. API Reference, 32. AI Hub,
33. Learning Resources.

These sections still import `antd` / `@ant-design/icons` /
`@heroicons/react` and are expected to be migrated in subsequent runs
of this branch. The ESLint rules continue to flag their files as
violations until they're migrated.

## Deps delta (so far)

### Added
- `react-hook-form`, `zod`, `@hookform/resolvers` — form stack.
- `sonner` — toast stack.
- `clsx`, `class-variance-authority`, `tailwindcss-animate` — shadcn deps.
- `eslint-plugin-litellm-ui` — local file: dep.
- `@radix-ui/*` transitive deps (pulled in by shadcn primitives).

### Removed
- `tailwind.config.js` (consolidated onto `.ts`).
- **Not yet removed** (pending final sweep at Task 45):
  `antd`, `@ant-design/icons`, `@heroicons/react`, `@remixicon/react`,
  `@headlessui/tailwindcss`. These remain in `package.json` as long as
  any un-migrated section still imports them.

## Bundle size delta

- **Pre-migration** (captured at end of Task 2, before any sections
  migrated): ~15.4 MB total JS across all `out/_next/static` chunks
  (see `docs/baseline-bundle-size.txt`).
- **Post-migration:** to be recorded at end of Task 45 (after deprecated
  deps are uninstalled — that is when the bundle-size win is realized).

## Known phase-2 prerequisites

Phase 1 deliberately did not touch:

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
- **DeleteResourceModal** (`src/components/common_components/
  DeleteResourceModal.tsx`) — shared chrome component used by many
  sections, still imports antd. To be migrated in Task 43 chrome sweep.

## Quirks and deviations

See `docs/QUIRKS.md` and `docs/DEVIATIONS.md` for per-section entries.

Key deviations:

- Playwright gates 4–5 (parity + snapshots) are **committed but not
  executed** in this cloud sandbox because neither the proxy nor a
  seeded dev-server auth helper is available. Specs are shape-complete
  and ready for reviewer execution.
- `DeleteResourceModal` and other shared chrome components are left for
  the final chrome sweep (Task 43).
- `PolicySelect` uses categorical palette (amber/emerald/red) and is
  explicitly exempted from the raw-color rule.
