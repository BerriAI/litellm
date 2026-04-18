# UI Dashboard (`ui/litellm-dashboard/`)

Rules for the Next.js admin UI served by the proxy.

## UI / Backend Consistency
- When wiring a new UI entity type to an existing backend endpoint, verify the backend API contract (single value vs. array, required vs. optional params) and ensure the UI controls match — e.g., use a single-select dropdown when the backend accepts a single value, not a multi-select

## UI Component Library
- **Always use `antd` for new UI components** — we are migrating off of `@tremor/react`. Do not introduce new `Badge`, `Text`, `Card`, `Grid`, `Title`, or other imports from `@tremor/react` in any new or modified file. Use `antd` equivalents: `Tag` for labels, plain `<span>`/`<div>` with Tailwind classes (or `Typography.Text`) for text, `Card` from `antd`, etc. Note that `antd` has no `"yellow"` Tag color — use `"gold"` for amber/yellow.

## Browser Storage Safety
- Never write LiteLLM access tokens or API keys to `localStorage` — use `sessionStorage` only. `localStorage` survives browser close and is readable by any injected script (XSS).
- Shared utility functions (e.g. `extractErrorMessage`) belong in `src/utils/` — never define them inline in hooks or duplicate them across files.

## MCP OAuth / OpenAPI (UI-side)
- `TRANSPORT.OPENAPI` is a UI-only concept. The backend only accepts `"http"`, `"sse"`, or `"stdio"`. Always map it to `"http"` before any API call (including pre-OAuth temp-session calls).
- FastAPI validation errors return `detail` as an array of `{loc, msg, type}` objects. Error extractors must handle: array (map `.msg`), string, nested `{error: string}`, and fallback.
