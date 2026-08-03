Never put LiteLLM tokens or API keys in `localStorage`. `localStorage` survives browser close. Prefer `httpOnly` cookies, or `sessionStorage` at most, understanding that any web storage is readable by injected scripts (XSS), and only httpOnly cookies are not

When you fix lint violations that are grandfathered in `eslint-suppressions.json`, run `eslint . --prune-suppressions` and commit the updated baseline so the gate ratchets down instead of leaving a stale suppression

`src/lib/http/schema.d.ts` is generated from the proxy's OpenAPI spec; never hand-edit it. After changing a backend route or response model that the dashboard consumes, run `npm run gen:api` and commit the result (CI `Check UI API Types Sync` enforces this)

Tests come in three tiers, named by the standard definitions. `Foo.test.tsx` is a unit test: one module, collaborators replaced by doubles, no multi-component tree, and it should run in milliseconds. `Foo.integration.test.tsx` renders a real component tree with real children and only stubs the network boundary; it costs seconds per case, so it earns its place by proving wiring that a unit test cannot reach. Browser-level tests live in `tests/e2e/ui/` as Playwright specs against a live proxy

When a component holds logic worth asserting, extract the logic and unit-test it there rather than driving it through a render. `CreateMCPServer` is the worked example: its payload building lives in `createServerPayload.ts` with 46 unit tests that run in single-digit milliseconds, while `CreateMCPServer.integration.test.tsx` keeps only the cases that prove a form field reaches the right payload key. A test that renders a whole modal to assert the shape of one object belongs in the first category, not the second

Most of the suite predates this split and is not yet classified, so an unsuffixed `*.test.tsx` is not evidence that a file is really a unit test. Classify what you touch
