# App Router migration smoke

A growing E2E smoke for pages migrated from the legacy `?page=` switch to App
Router path routes. For each page it clicks the sidebar link by its accessible
name, verifies the destination's content, reloads it, then visits Virtual Keys
and returns. It runs at the default mount and a non-root `SERVER_ROOT_PATH` mount

Link selection does not depend on `href` formatting. URL assertions compare the
origin and pathname, allowing a trailing slash, query string, and fragment while
rejecting another route or mount. Reloads must return a successful document,
and each journey must finish without uncaught browser errors

## Adding a page

Add an entry to `tests/e2e/ui/fixtures/migratedPages.ts`, keyed by the legacy page
ID. Specify its route segment, accessible link name, sidebar group if collapsed,
and distinctive visible content such as a heading or tab. Keep expectations
independent of the application's route table so an incorrect destination fails
the test. Both navigation suites use this fixture

For a licensed-only page, `unlicensedText` describes the expected upgrade notice.
The authenticated session's license claim determines which content must render

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
tests. Under a non-root mount it logs in at `${SERVER_ROOT_PATH}/ui/login`

`tests/navigation/sidebar.spec.ts` also checks the navigation helpers against
equivalent link formats on the live dashboard and a deep link containing a query
string and fragment. The link-format cases change only the rendered `href`
attribute to exercise the locator contract; destination pages and APIs remain live
