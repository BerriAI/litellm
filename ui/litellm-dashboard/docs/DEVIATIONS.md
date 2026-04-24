# Blueprint deviations and intentional snapshot rebaselines

Format per entry:

## <section> — <short title>

- **Type:** `blueprint deviation` | `snapshot rebaseline` | `dead-code removal` | `blueprint update — stress test`
- **Blueprint rule deviated from (if applicable):** ...
- **What the agent did instead:** ...
- **Why:** ...

---

## Section 1 (Access Groups) — gate 4 + 5 skipped (Playwright parity + snapshots)

- **Type:** environmental-gate skip
- **Blueprint rule deviated from:** `docs/RECIPE.md` steps 7–8 (Playwright
  parity spec + visual snapshot baseline).
- **What the agent did instead:** Committed the parity spec file
  (`e2e_tests/parity/access-groups.spec.ts`) with full shape-complete
  assertions and a visual snapshot call, but **did not run** the spec.
  Relied on TS + Lint + Vitest + build gates for correctness.
- **Why:** The cloud sandbox has no running proxy at port 4000 and no
  seeded dev-server auth helper (phase 1 has not yet wired one). Running
  Playwright against a cold UI without auth produces 0 passing assertions,
  which is worse than skipping. Later runs or the human reviewer will
  execute the spec once a dev-server auth helper (or the proxy stack) is
  available. The same gate-skip applies to all subsequent sections for the
  duration of this autonomous run.

---

## Section 1 (Access Groups) — DeleteResourceModal deferred to chrome sweep

- **Type:** blueprint deviation
- **What the agent did instead:** `DeleteResourceModal`
  (`src/components/common_components/DeleteResourceModal.tsx`) still
  imports antd. It is a shared component used by multiple sections.
  Leaving it for the global chrome sweep (Task 43) to migrate in a single
  coordinated pass, so we don't thrash the component once per owning
  section.
- **Why:** Shared-chrome component with no section that exclusively owns
  it. The RECIPE.md explicitly allows deferring shared parent components
  that aren't covered by the current section's parity spec.

---

## Section 6 (Tool Policies) — categorical colors (amber/emerald/red) on policy badges

- **Type:** `blueprint deviation`
- **Blueprint rule deviated from:** §1 "semantic Tailwind tokens only".
- **What the agent did instead:** `PolicySelect.tsx` uses raw palette
  classes (`bg-amber-100 text-amber-800 border-amber-300`,
  `bg-emerald-100 ...`, `bg-red-100 ...`) and is added to the
  `litellm-ui/no-raw-tailwind-colors` override list in `.eslintrc.json`.
- **Why:** The policy badges encode a **categorical** (untrusted /
  trusted / blocked), not a theme state. Semantic tokens (`primary`,
  `muted`, `destructive`, ...) don't have three distinct hues suitable
  for three categorical values. A phase-2 task could introduce semantic
  policy tokens (`--policy-untrusted`, `--policy-trusted`,
  `--policy-blocked`) backed by CSS variables; out of scope for phase 1.

## Users section — categorical colors (red/green/blue/yellow) in bulk-create + edit status UI

- **Type:** `blueprint deviation`
- **Blueprint rule deviated from:** §1 "semantic Tailwind tokens only".
- **What the agent did instead:** `bulk_create_users_button.tsx`,
  `BulkEditUsers.tsx`, `DefaultUserSettings.tsx`, and
  `view_users/user_info_view.tsx` keep raw palette classes
  (`bg-red-50`/`text-red-500`, `bg-green-100`/`text-green-500`,
  `bg-blue-50`/`text-blue-500`, `bg-yellow-50`/`text-yellow-700`,
  `bg-blue-100` model chips, `text-amber-600` warning callout, etc.) and
  are added to the `litellm-ui/no-raw-tailwind-colors` override list in
  `.eslintrc.json`.
- **Why:** The CSV import flow uses red = invalid / failed, green =
  success, blue = info / pending, yellow = structural warning — a
  categorical, status-driven palette with four distinct hues. Only
  `destructive` maps cleanly; `primary`/`muted` don't cover "success" or
  "info/pending" as separate categories. A phase-2 task could introduce
  semantic status tokens (`--status-success`, `--status-info`,
  `--status-warning`, `--status-failure`); out of scope for phase 1.

## Section 1 (Access Groups) — custom MultiSelect shim in lieu of antd Select mode=multiple

- **Type:** `blueprint deviation`
- **Blueprint rule deviated from:** "use shadcn `Select`" for multi-select.
- **What the agent did instead:** introduced a small in-file `MultiSelect`
  wrapper (shadcn `Select` + chip list rendered below) in
  `AccessGroupBaseForm.tsx`. The shadcn primitive does not ship a multi-
  select mode; this is the canonical phase-1 substitute. Recorded in
  `BLUEPRINT.md §1` so subsequent sections use the same shape.
- **Why:** No multi-select shadcn primitive exists; building a dedicated
  `MultiSelect` component in `@/components/ui/` is phase-2 work. The
  inline wrapper is small enough to copy/paste into any section that
  needs it until phase 2.

