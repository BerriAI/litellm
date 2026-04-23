# Phase-1 parity specs

One Playwright spec file per migrated section: `<section>.spec.ts`.

## Conventions

- **Assert on user-observable behavior only:** `getByRole`, `getByText`,
  `getByLabel`, `getByTestId`. Never assert on DOM selectors that identify the
  framework (e.g. `.ant-btn`).
- **Visual snapshots** live in `<section>.spec.ts-snapshots/` next to each
  spec. `expect(page).toHaveScreenshot()` is the canonical call.
- **Both old and new UI pass:** specs must pass against the current UI
  (sanity) before the section is migrated, and against the migrated UI after.
- **Exception:** when a `QUIRKS.md` entry uses resolution `fix`, the spec
  captures the corrected behavior; the old-UI sanity check is explicitly
  skipped (record the skip in `QUIRKS.md`).
- **Soft cap:** ~30 assertions per spec. If the section genuinely needs more,
  split into `<section>-a.spec.ts` / `<section>-b.spec.ts`.

## Running

Default project target is the local Next dev server at
`http://localhost:3000` (started automatically by `webServer` in
`playwright.config.ts`):

```bash
npx playwright test --project=parity e2e_tests/parity/<section>.spec.ts
```

To run parity specs against the proxy instead (e.g. because dev-server auth
mocking is too brittle for the section), set `PARITY_BASE_URL`:

```bash
PARITY_BASE_URL=http://localhost:4000 \
  npx playwright test --project=parity e2e_tests/parity/<section>.spec.ts
```

When `PARITY_BASE_URL` is set, the dev-server `webServer` hook does not
start; you are responsible for having the target server running.

## Snapshot baselining

First run per section: create baselines.

```bash
npx playwright test --project=parity \
  e2e_tests/parity/<section>.spec.ts --update-snapshots
```

Subsequent runs (verification): no `--update-snapshots`. Pixel diffs count as
regressions. Intentional rebaselines are logged in `docs/DEVIATIONS.md`.

## Authentication

Dev-server auth is a known weak spot for phase-1 parity. Options per spec:

1. **Playwright route interception** — stub the `/get/*` endpoints the UI hits
   during boot (`get/ui_theme_settings`, `get/sso_settings`, etc.) + seed
   `sessionStorage` with a fake JWT so useAuthorized returns admin.
2. **Use `PARITY_BASE_URL=http://localhost:4000`** — run against the real
   proxy with a seeded admin user. Slower but more realistic.

Pick per section depending on how much the section exercises backend calls.
