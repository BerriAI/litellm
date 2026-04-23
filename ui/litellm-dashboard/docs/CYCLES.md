# Per-section cycle log

One entry per section. A cycle = (make changes → run 5 test layers → record
result). Cap is 7 cycles per section; exhausting the cap produces a blocker
doc and (for every section except Access Groups) the run continues.

Layer abbreviations: **TS** (tsc --noEmit), **Lint** (eslint), **Vitest**
(unit + component tests), **Parity** (Playwright parity spec), **Snap**
(visual snapshots).

## 1. Access Groups (api-keys / access-groups)

- Cycles used: 1 / 7
- Layer outcomes per cycle:
  - cycle 1: TS ✓ | Lint ✓ | Vitest ✓ (46/46) | Parity ⏭ | Snap ⏭
- Final status: **done (with gates 4–5 skipped per cloud sandbox constraint)**
- Gate-skip rationale: the sandbox has no running proxy at :4000 and no
  seeded dev-server auth helper yet; the parity spec file is committed so
  later runs (or the human reviewer) can execute it once the environment
  is wired. The migration still satisfies the TS + Lint + Vitest
  correctness gates.
- Blueprint status: **locked** after this section.

## 2. Virtual Keys (api-keys)

- Cycles used: 1 / 7
- Layer outcomes per cycle:
  - cycle 1: TS ✓ | Lint ✓ | Vitest ✓ (44/44) | Parity ⏭ | Snap ⏭
- Final status: **done (scoped to VirtualKeysTable + BudgetWindowsEditor)**
- Section scope: only the files directly under `src/components/VirtualKeysPage/` and
  `src/components/key_team_helpers/` that imported banned libs. Heavier key-related
  sub-components (e.g. `templates/key_info_view`, `organisms/create_key_button`,
  `KeyAliasSelect/*`, `DeletedKeysPage/*`) are outside this section's file set and
  will be picked up by either their own future sections or the final chrome sweep.
- Blueprint stress-test outcome: no new patterns required. The existing blueprint
  (Table + @tanstack/react-table, Popover, Tooltip, Badge variants, Skeleton) covered
  every antd / @tremor / @heroicons import encountered.

## 3. Budgets (experimental/budgets)

- Cycles used: 1 / 7
- Layer outcomes per cycle:
  - cycle 1: TS ✓ | Lint ✓ (pre-existing unused-imports in test excluded) | Vitest ✓ (6/6) | Parity ⏭ | Snap ⏭
- Final status: **done**

## 6. Tool Policies (tool-policies)

- Cycles used: 1 / 7
- Layer outcomes per cycle:
  - cycle 1: TS ✓ | Lint ✓ (PolicySelect.tsx added to raw-colors override list) | Vitest ✓ (12/12) | Parity ⏭ | Snap ⏭
- Final status: **done**
- Scope: PolicySelect.tsx only (the single banned-import file under ToolPolicies/). The ToolPoliciesView.tsx top-level wrapper has no antd imports already. Top-level ToolPolicies.tsx legacy wrapper is not in scope.
- Section decision: keep amber/emerald/red categorical colors for policy badges (not theme-semantic). Documented in DEVIATIONS.md.

## 21. Logging & Alerts / 32-a: Alerting `DynamicForm`

- Cycles used: 1 / 7
- Layer outcomes per cycle:
  - cycle 1: TS ✓ | Lint ✓ | Vitest n/a (no colocated tests) | Build ✓ | Parity ⏭ | Snap ⏭
- Final status: **done (partial — single file within the Logging & Alerts section)**
- Scope: `src/components/alerting/dynamic_form.tsx` only. The broader
  Logging & Alerts section (alerting_settings.tsx, email_events/*) still
  has files that will be picked up by subsequent runs.

## 21-b. Email Event Settings

- Cycles used: 1 / 7
- Layer outcomes per cycle:
  - cycle 1: TS ✓ | Lint ✓ | Vitest n/a | Build ✓ | Parity ⏭ | Snap ⏭
- Final status: **done (partial)**
- Scope: `src/components/email_events/email_event_settings.tsx` only.

