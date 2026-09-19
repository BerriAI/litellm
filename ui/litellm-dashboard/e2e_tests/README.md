# UI End-to-End Tests

Playwright tests that drive the LiteLLM dashboard against a real, fully built
proxy. They started life as the manual UI QA checklist and now run on every
build as the CircleCI `e2e_ui_testing` job

## What these test, and what they don't

The guiding principle is to test the bundle we actually ship. A run builds the
Next.js UI from source, serves it from the proxy exactly as production does,
and exercises it through a real browser against a real proxy and a real
Postgres. The only faked dependency is the LLM itself: `fixtures/mock_llm_server/server.py`
returns canned streaming and non-streaming completions so tests stay
deterministic and free. Mock the boundary that costs money or flakes; keep
everything else real

## Running

From this directory:

```bash
./run_e2e.sh                 # full run: boots postgres, mock LLM, proxy, seeds, runs Playwright
./run_e2e.sh --headed        # watch it drive the browser
./run_e2e.sh --repeat-each=5 # flake hunting
SLOWMO=250 ./run_e2e.sh --headed   # slow each action down for debugging
```

Local runs need `docker`, `psql`, `uv`, and `npx` on PATH. The script spins up
its own throwaway Postgres container and cleans everything up on exit. In CI
(`CI=true`) it expects Postgres, `DATABASE_URL`, and the toolchain to already be
present and skips the container

The `SERVER_ROOT_PATH` redirect spec is incompatible with the shared login
setup, so it lives behind its own `serverRootPath.config.ts` and runs in the
`test_server_root_path` GitHub workflow

## How a run works

The proxy serves both the API and the static UI from `:4000`, so there is one
origin and no CORS to reason about. `run_e2e.sh` builds the UI, copies the
export into `litellm/proxy/_experimental/out`, and restructures `login.html`
into `login/index.html` so extensionless routes resolve the way the proxy
serves them. It then boots the mock LLM on `:8090` and the proxy on `:4000`,
pushes the Prisma schema, applies `fixtures/seed.sql`, and finally runs
Playwright. `globalSetup.ts` logs each seeded role in once through the real
login form and writes a `storageState.json` per role; specs reuse those
sessions instead of logging in over and over

## Layout

```
fixtures/
  seed.sql            deterministic, idempotent test data (all rows prefixed e2e-)
  config.yml          proxy config: two fake models pointed at the mock LLM
  mock_llm_server/    FastAPI server returning canned OpenAI-shaped responses
  users.ts            role -> credentials + storage-state path
  pages.ts            Page enum: the ?page= values the app routes on
  menuMappings.ts     sidebar label -> Page, for nav assertions
constants.ts          shared IDs/aliases that must match seed.sql
globalSetup.ts        one login per role, saved as storage state
helpers/navigation.ts navigateToPage, popup dismissal, shared row helpers
tests/<role-or-area>/ specs grouped by role and feature
```

## Writing a new test

Pick the role you are testing as and load its session with
`test.use({ storageState: ADMIN_STORAGE_PATH })`. Reference seeded data through
`constants.ts` rather than hardcoding IDs, and never depend on data another test
created

Find elements through the accessibility tree first: `getByRole`,
`getByPlaceholder`, `getByTestId`. Ant Design's internal CSS classes are not a
stable API, so only reach for selectors like `.ant-modal:visible` when nothing
better exists, and scope them to a container (`modal.getByRole(...)`) so strict
mode stays happy

Assert on what the user observes (a success toast, a row appearing, the URL),
and when the bug you care about lives below the surface, prove it: capture the
outgoing request with `page.waitForRequest` or re-check the management API.
`tests/modelsPage/clearCustomPricing.spec.ts` is the reference example; it
drives the form, asserts the PATCH body carries explicit nulls, then confirms
the override is gone from the DB

Never sleep. Arm a wait before the action that triggers it
(`Promise.all([page.waitForURL(...), button.click()])`) and wait on the specific
response you depend on rather than on first paint. The suite runs fully
parallel, so give created resources unique names (a `Date.now()` suffix) and put
shared state back the way you found it in `afterEach` or a `try/finally`

Tests that need an env-dependent proxy (a license, an external logout URL)
should `test.skip` when the var is absent so they pass on a laptop, while CI
sets the var and the test fails loudly if the wiring ever breaks

Comments here earn their place by explaining a race, an Ant Design quirk, or why
a workaround exists, not by narrating what the next line does

## Debugging a failure

CI stores the Playwright HTML report and traces as artifacts (`trace` is
captured on first retry). Locally, `--headed` plus `SLOWMO` lets you watch the
run, and `globalSetup` drops a screenshot under `test-results/` if login itself
fails
