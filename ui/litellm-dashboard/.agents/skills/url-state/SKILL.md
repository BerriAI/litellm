---
name: url-state
description: Read or write dashboard URL query state (tabs, filters, pagination, deep links, demo flags) with nuqs instead of raw history or URLSearchParams
---

# URL state

Query params the dashboard reacts to go through nuqs: `useQueryState` or `useQueryStates` with parsers from `"nuqs"`. Do not call `window.history.pushState` or `replaceState`, hand-edit a `URLSearchParams`, or parse `useSearchParams().get(...)` for that state. Raw writes bypass nuqs, so other hooks on the same key go stale and queued nuqs updates can overwrite them

Validate in the parser, not after reading. Use `parseAsStringLiteral` for enums, `parseAsInteger`, `parseAsBoolean`, `parseAsArrayOf` for lists, `.withDefault` for defaults, and `createParser` when the wire format is custom (for example `demo=1`). Keep the parser map in a module const and share it between every reader and writer of the same key

Clear a param with `setter(null)`; nuqs keeps unrelated params and the hash. Updates replace history by default, pass `{ history: "push" }` when the change should be a back-button step. A param consumed once on arrival (a deep link or OAuth return flag) is captured in a `useState` initializer from the hook value and then cleared with the setter in a mount effect

Raw reads are fine for one-time reads that never re-render on URL changes: OAuth callback pages, login, the legacy `?page=` redirect, and `networking.tsx`

Reuse the existing helpers before adding a hook: `useUrlTab` in `src/hooks/useUrlTab.ts`, `useUrlTableState` in `src/components/shared/DataTable`, and the route modules `src/components/lens/route.ts` and `src/components/logs/request/logDetailRouting.ts`

In tests, render with `renderWithProviders` from `tests/test-utils.tsx`, passing `searchParams` for the initial URL and `onUrlUpdate` to assert writes. When a test asserts `window.location` directly, wrap the render in `NuqsAdapter` from `nuqs/adapters/react`; writes flush asynchronously, so assert them inside `waitFor`. For inputs bound to URL state, set values with one `fireEvent.change` rather than `user.type`, which flakes when a re-render between keystrokes moves focus

See the [nuqs docs](https://nuqs.dev/docs) for parser and option details
