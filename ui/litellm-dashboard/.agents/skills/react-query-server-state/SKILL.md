---
name: react-query-server-state
description: Design, migrate, audit, and test TanStack Query server state in the LiteLLM dashboard. Use for query keys, initial readiness, filter changes, cached results, retries, cancellation, and replacing manual fetch lifecycle tracking
---

# Dashboard server state

Read the installed `@tanstack/react-query` version and the relevant [React Query guides](https://github.com/TanStack/query/tree/main/docs/framework/react) before deciding behavior. Start with queries, network mode, disabling queries, query keys, important defaults, and testing. Ask the user about unresolved product requirements after reading the docs

Use the existing provider and query cache for reads. Mutations and deliberate imperative actions have different lifecycles. Request IDs, serialized dependency keys, cancellation flags, copied results, and separate settled/error state around a read usually duplicate query behavior. Remove them when migrating unless they encode a real product requirement

`hooks/dailyActivity/dailyActivityQueries.ts` provides a local example of shared keys and options. Keep request identity with the fetcher, using an options factory when a parent supplies a query. A bare callback does not describe which data belongs to the current filters

## Review decisions

Determine whether two requests can produce different responses. Include their varying inputs in the key, including authorization scope, entity, dates, filters, model, and pagination. Plain serializable objects work; object property order does not change identity. Do not serialize keys manually or use callback identity as request identity. Changes to authorization must not expose another session's cached results. Review key factories and invalidation prefixes together, since lint cannot infer dependencies hidden inside helpers

Separate initial readiness from network activity. `isQueryPending(query)` in `hooks/common/queryReadiness` checks `isEnabled && isPending`. An enabled offline query with no data is pending while its fetch is paused; `isLoading` and `isFetching` are both false. A disabled query without data is idle for the UI. Cached data remains available during a paused refresh. Use `isFetching` for an activity indicator, and `isPaused` when an explicit offline message helps the user

Decide whether disabled means hidden, unavailable, or waiting for another input. The readiness helper does not make that product decision. It must not turn missing required inputs into a successful zero result

Decide whether previous data is valid under a new filter. Usage totals must not silently show another entity or date range. Avoid `keepPreviousData` there. Pagination can benefit from previous data if the UI makes `isPlaceholderData` clear. A cached result for the exact same key can render immediately, but freshness is a separate decision. Set `staleTime` according to how quickly that endpoint changes and the cost of refetching, rather than copying a value from another endpoint

Preserve errors as errors. Query functions return data or reject; catching an error and returning an empty array makes failure look like success. Distinguish initial failure from background refresh failure when cached data exists. Retry uses `refetch`; decide whether an error retry should replace the error display with a loading indicator. `skipToken` disables safely but does not support an imperative refetch, so choose `enabled` when a manual retry of that disabled query is required

Prefer `select` for derived query data. Keep expensive selectors stable. Avoid rest destructuring query results because it subscribes to every tracked property. Dependency arrays should contain the fields used, not the entire query result

Pass the query function's `AbortSignal` through networking helpers that support it. Query keys isolate late responses even without transport cancellation. Cancellation saves work but does not replace correct identity. Invalidate relevant key prefixes after writes, including any authorization segment shared with readers

## Behavioral verification

Mock the network boundary and use the real query hook for regressions. A hook mock cannot prove cache identity, offline behavior, or enabled logic. Existing presentation tests may mock hooks, but their pending and enabled fields must reflect the state they claim to simulate

Use a fresh query client per test and disable retries for deterministic error tests. Clear the shared client when using `renderWithProviders`, whose defaults intentionally retain cached results. Restore global `onlineManager` state after offline tests

Exercise the failure mode rather than implementation structure. Cover an enabled offline request with no data, reconnection, disabled inputs, and a cached paused refresh when changing readiness behavior. For filter identity, resolve an old request after a newer request and assert the old data never replaces the current view. When clearing a filter restores cached data, use distinct filtered and unfiltered results and assert the restored result. A passing spinner assertion and an unchanged request count do not prove restoration

React Query tracks accessed result properties. When testing a status transition with `renderHook`, access the property under assertion before triggering the transition, or deliberately subscribe with `notifyOnChangeProps` in the test. Otherwise an unobserved status change can leave the test's result stale

Run the affected tests and the dashboard lint commands. Use live browser QA for user-visible behavior when the proxy and dashboard are available. Unit tests establish regression behavior; report live QA and hosted CI separately, and never describe missing reviews as passing

## Enforcement

`eslint.config.mjs` loads the official [TanStack Query recommended flat config](https://tanstack.com/query/latest/docs/eslint/eslint-plugin-query). It checks visible key dependencies, stable clients, unstable hook dependencies, property ordering, rest destructuring, and void query functions

The local typed rule `local/no-query-is-loading` rejects native query fetch flags used as initial readiness across custom hooks and aliases. Keep this rule and the behavioral regressions: the official plugin does not enforce the dashboard's offline readiness contract. Existing CI runs ESLint and its budgets, so new errors fail there without a separate lint runner

Lint is evidence about supported syntax, not proof of product correctness. Review custom wrappers, helper-generated keys, stale placeholders, and failure/empty rendering paths explicitly. Keep deterministic checks in lint or tests and judgment guidance in this skill
