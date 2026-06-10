# App Router migration smoke

A growing E2E smoke for pages migrated from the legacy `?page=` switch to App
Router path routes. For each migrated page it clicks the page's sidebar link, checks
the URL is the path route and the page renders, reloads it, then clicks off to a
legacy page and back to confirm navigation still works. It runs in two situations:
the default mount and a non-root `SERVER_ROOT_PATH` mount.

## Adding a page

When a page's migration merges, add its route segment to
`e2e_tests/fixtures/migratedPages.ts` (keep it in lockstep with `MIGRATED_PAGES`
in `src/utils/migratedPages.ts`). Both suites pick it up automatically.

## Running

Build the UI into the proxy and start the proxy first (the suite runs against
`http://localhost:4000`).

Default mount:

```
npm run e2e:migration
```

Non-root mount (build and boot the proxy with the same root path, e.g. `/litellm`):

```
SERVER_ROOT_PATH=/litellm npm run e2e:migration:root
```

`globalSetup` logs in once per role; the admin storage state is reused for these
tests. Under a non-root mount it logs in at `${SERVER_ROOT_PATH}/ui/login`.
