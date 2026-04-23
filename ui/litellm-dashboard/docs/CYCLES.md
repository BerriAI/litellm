# Per-section cycle log

One entry per section. A cycle = (make changes → run 5 test layers → record
result). Cap is 7 cycles per section; exhausting the cap produces a blocker
doc and (for every section except Access Groups) the run continues.

Layer abbreviations: **TS** (tsc --noEmit), **Lint** (eslint), **Vitest**
(unit + component tests), **Parity** (Playwright parity spec), **Snap**
(visual snapshots).

## <section>

- Cycles used: N / 7
- Layer outcomes per cycle:
  - cycle 1: TS ✓ | Lint ✗ | Vitest — | Parity — | Snap —
  - cycle 2: TS ✓ | Lint ✓ | Vitest ✓ | Parity ✓ | Snap ✓
- Final status: done | blocked | wip
