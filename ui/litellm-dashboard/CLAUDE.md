Never put LiteLLM tokens or API keys in `localStorage`. `localStorage` survives browser close. Prefer `httpOnly` cookies, or `sessionStorage` at most, understanding that any web storage is readable by injected scripts (XSS), and only httpOnly cookies are not

When you fix lint violations that are grandfathered in `eslint-suppressions.json`, run `eslint . --prune-suppressions` and commit the updated baseline so the gate ratchets down instead of leaving a stale suppression

`src/lib/http/schema.d.ts` is generated from the proxy's OpenAPI spec; never hand-edit it. After changing a backend route or response model that the dashboard consumes, run `npm run gen:api` and commit the result (CI `Check UI API Types Sync` enforces this)
