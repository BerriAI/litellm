# Section migration recipe (phase 1 shadcn)

Every section in Part C of the phase-1 plan follows this recipe exactly.
Section-specific tasks list only the section label, its leftnav page key,
the known quirks, and any deviations from the recipe.

## Inputs

- **Section label** — e.g. "Access Groups".
- **Leftnav page key** — the string used in `src/components/leftnav.tsx`.
- **Current-UI entrypoint** — discovered via discovery commands below.

## Discovery

Before any edits, run these commands and record the results in the section's
scratch notes (kept in `docs/blockers/_scratch/<section>.md`; scratch files
are not committed — only blocker docs are).

```bash
# 1. Find the primary route file (not every section has one; some render via
#    a legacy page-switch inside the dashboard layout).
find ui/litellm-dashboard/src/app -name "page.tsx" -path "*<section-slug>*"

# 2. Find the primary component directory.
find ui/litellm-dashboard/src/components -type d -iname "*<section-keyword>*"

# 3. List files with banned imports within the section's scope.
grep -rlE "from ['\"](antd|@ant-design/icons|@heroicons/react|@remixicon/react|@tremor/react)" \
  ui/litellm-dashboard/src/components/<section-dir>/ \
  ui/litellm-dashboard/src/app/\(dashboard\)/<section-route>/ 2>/dev/null
```

## Step-by-step

### 1. Author the parity spec against the current UI

Start the current app (`npm run dev`) and use Playwright MCP (or manual
exploration) to walk the section. Identify every user-observable behavior:
labels, form fields, error messages, success toasts, filters, sort controls,
pagination, empty state, loading state, permissions gates.

Write `e2e_tests/parity/<section>.spec.ts` asserting via semantic queries
(`getByRole`, `getByText`, `getByLabel`, `getByTestId`). **Never** framework
selectors like `.ant-btn`.

Minimum coverage per section:
- list renders
- empty state
- create happy path
- edit happy path
- delete with confirm
- one representative filter
- permissions gate (non-admin view)
- error toast on backend failure (mock 500 via `page.route`)

Soft cap ~30 assertions per spec. Split into `<section>-a.spec.ts` /
`<section>-b.spec.ts` if needed.

### 2. Sanity check — parity spec passes against current UI

```bash
cd ui/litellm-dashboard && npx playwright test --config e2e_tests/playwright.config.ts \
  --project=parity e2e_tests/parity/<section>.spec.ts
```

If this fails, the spec is wrong (or the section hits a known quirk) — fix
the spec, not the UI. Current UI passing the spec is the sanity-check step.

**Exception:** if the current UI has a quirk you've decided to *fix* in
migration, add a `QUIRKS.md` entry and mark the offending assertion with a
comment `// quirk: fix — skip sanity check`. Skip running that assertion
against the current UI.

### 3. Migrate the section

For each file in the discovery list:

- Replace `antd` imports with shadcn equivalents from `@/components/ui/*`.
  See the translation table in `BLUEPRINT.md` (drafted in Section 1; locked
  thereafter).
- Replace `@ant-design/icons`, `@heroicons/react`, `@remixicon/react` imports
  with `lucide-react`.
- Replace `antd.message.*` / `antd.notification.*` call sites → `MessageManager`
  / `NotificationManager` (these now delegate to sonner). Most call sites
  already use the managers; only check for direct-antd-import stragglers.
- Replace raw Tailwind color classes with semantic shadcn tokens
  (`bg-primary`, `text-foreground`, `bg-muted`, `border-border`, etc.).
- For forms: rewrite using `react-hook-form` + `zod` + shadcn `<Form>`.
  Colocate zod schemas in the same file if small, or in
  `<form-component>.schemas.ts` if larger. Submit handler continues to call
  the existing data-layer function unchanged.

For shared parent components a section depends on (e.g. `Sidebar2.tsx`,
`Navbar.tsx`): migrate them **only if** the section's parity spec covers
them. Otherwise leave for a later section that owns them — the final global
chrome sweep (Task 43) will catch any stragglers.

### 4. Gate layer 1 — TypeScript

```bash
cd ui/litellm-dashboard && npx tsc --noEmit
```

Must pass for files the agent touched. Pre-existing errors in test files
from other sections are ignored.

### 5. Gate layer 2 — Lint (banned-imports + no-raw-colors)

```bash
cd ui/litellm-dashboard && npm run lint
```

Must pass for files the agent modified in this section. Pre-existing
violations in un-migrated sections are expected and ignored — but all
violations in the section's own file set must be zero.

### 6. Gate layer 3 — Vitest + RTL

Every form migrated must have an RTL test (`*.test.tsx` colocated with the
form). Tests must render, validate (attempt invalid input, assert error
text), and submit (assert handler called with expected data).

All existing RTL tests must continue to pass. If a test uses an antd-
specific selector (e.g. `.ant-input`), update it to a semantic query
(`getByRole("textbox", { name: /.../ })`) — do not delete the test. If a
test mocks `antd.message.*` directly, update the mock to point at
`MessageManager` from `@/components/molecules/message_manager`.

```bash
cd ui/litellm-dashboard && npx vitest run
```

Must pass.

### 7. Gate layer 4 — Playwright parity spec against migrated UI

```bash
cd ui/litellm-dashboard && npx playwright test --config e2e_tests/playwright.config.ts \
  --project=parity e2e_tests/parity/<section>.spec.ts
```

Must pass.

### 8. Gate layer 5 — Visual snapshot baseline

First run for a section: create the baseline.

```bash
npx playwright test --config e2e_tests/playwright.config.ts \
  --project=parity e2e_tests/parity/<section>.spec.ts --update-snapshots
```

Subsequent runs: no `--update-snapshots` — pixel diffs count as regressions.

### 9. Append to `CYCLES.md`

Record cycle count + final status (done / blocked / wip).

### 10. Commit

- **If section succeeded:**

  ```bash
  git add ui/litellm-dashboard/
  git commit -m "feat(ui): migrate <section> to shadcn"
  ```

- **If section blocked after 7 cycles:**
  - Write `docs/blockers/<YYYY-MM-DD-HHMMSS>-<section>.md` with what
    failed, what was tried, suspected root cause.
  - Commit any partial work with a `wip(ui):` prefix.
  - **Move on to the next section.** Do not retry. The one exception is
    Section 1 (Access Groups): if it fails, halt the entire run.

## Cycle accounting

- A cycle = one attempt at the full gate sequence (steps 4–8).
- Cap is 7 per section.
- Cycles are logged in `CYCLES.md` as they happen.

## Quirks

- If a quirk is found during step 1 (parity spec authoring), log it in
  `QUIRKS.md` **before** writing the assertion that would capture or skip it.

## Deviations

- If blueprint rules don't fit a section (e.g. the section has a flow
  canvas and can't use standard table patterns), log the deviation in
  `DEVIATIONS.md` **before** acting on it.
