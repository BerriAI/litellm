---
name: react-query-server-state
description: Fetch server state in the LiteLLM dashboard with TanStack React Query instead of hand-rolled useEffect/useState request tracking. Use when adding or reviewing any hook or component that calls a networking function, when you see request IDs, `cancelled` flags, `settled` state, serialized dependency keys, or separate `loading`/`failed` state next to a fetch, or when writing tests for components that call `useQuery`.
---

# Server state with React Query

The dashboard already ships `@tanstack/react-query` v5 and wraps every page in `src/contexts/ReactQueryProvider.tsx`. Any data that comes from the proxy is server state and belongs in a query. Reference docs: https://github.com/TanStack/query/tree/main/docs/framework/react (start with `guides/important-defaults.md`, `guides/query-keys.md`, `guides/queries.md`, `guides/disabling-queries.md`, `guides/testing.md`).

## The anti-pattern

If a hook or component contains any of the following, it is reimplementing React Query badly and should be migrated:

```ts
const [settled, setSettled] = useState<{ key: string; data: T; failed: boolean } | null>(null);
const requestIdRef = useRef(0);
const depsKey = JSON.stringify(deps);
useEffect(() => {
  const requestId = ++requestIdRef.current;
  fetchSomething().then((r) => { if (requestIdRef.current === requestId) setSettled(...) });
  return () => { requestIdRef.current++; };
}, [depsKey]);
const loading = enabled && settled?.key !== depsKey;
```

Request IDs, `cancelled` flags, "settled for which key" bookkeeping, retry tokens, and parallel `loading`/`failed` booleans are exactly what the query cache does: one cache entry per key, stale responses for an old key never reach the new key, status is derived, retries and refetches are built in.

## The pattern

Put keys and options in one place so every caller and every invalidation agrees on identity. The daily activity queries in `src/app/(dashboard)/hooks/dailyActivity/dailyActivityQueries.ts` are the reference implementation.

```ts
import { queryOptions, skipToken, useQuery } from "@tanstack/react-query";

export const thingKeys = {
  all: ["thing"] as const,
  detail: (request: ThingRequest | null) => [...thingKeys.all, "detail", request] as const,
};

export const thingQueryOptions = (request: ThingRequest) =>
  queryOptions({
    queryKey: thingKeys.detail(request),
    queryFn: () => fetchThing(request),
  });

export const useThing = (request: ThingRequest | null) =>
  useQuery({
    queryKey: thingKeys.detail(request),
    queryFn: request ? () => fetchThing(request) : skipToken,
  });
```

Rules that matter:

1. The query key contains every value the query function reads: access token, entity type, ids, date range, filters, model, page. If two calls can return different data, their keys must differ. Plain objects, arrays, strings, numbers, null, and `Date` (serialized via `toJSON`) are all fine in a key; object key order does not matter. Never put a function in a key and never compare callback identity to decide whether data is current. If a component receives "how to fetch" from its parent, pass a `queryOptions` factory (for example `(model: string) => QueryOptions`) rather than a bare `() => Promise<T>`, so the key travels with the fetcher.
2. Do not hand-serialize keys with `JSON.stringify`; React Query hashes keys deterministically.
3. Disable with `skipToken` (type safe) or `enabled: false` when inputs are missing. A disabled query with no cache is `isPending` but not fetching, so use `isLoading` (`isPending && isFetching`) for "show a spinner", not `isPending`.
4. The query function must return data or throw. Networking helpers in `src/components/networking.tsx` already throw on non-2xx, so do not wrap them in `try/catch` that returns an empty value; that turns errors into successful empty data and hides them. Do not return `undefined`.
5. Derive UI state from the result: `isLoading` for first load, `isError`/`error` for failure, `data ?? EMPTY` for rendering, `isFetching` only for a background refresh indicator. Do not copy query state into `useState`, and do not wrap the result in a custom `{ loading, failed }` object unless an existing public interface needs it, in which case map directly (`loading: query.isLoading, failed: query.isError`).
6. Transform with `select` instead of `useEffect` + `setState`. Keep `select` referentially stable (module level function or `useCallback`), otherwise it reruns every render.
7. Retry buttons call `query.refetch()`. While a refetch from the error state is running, `isError && isFetching` is true; render that as loading if the old UI did.
8. Do not reach for `placeholderData: keepPreviousData` by default. Several dashboard views (usage, cost optimization) must never show numbers from the previous date range or entity under the new filter; a fresh key with no cache renders the empty or loading state, which is the correct behaviour there. Opt in only when showing stale data during a key change is explicitly desired, and then surface `isPlaceholderData`.
9. Defaults: `staleTime` is 0, refetch on mount, window focus, and reconnect, three retries with exponential backoff, and inactive cache kept for five minutes. Expensive aggregate endpoints should set a `staleTime` (the daily activity queries use a shared constant) so tab switches do not re-run them.
10. Cancellation: React Query passes an `AbortSignal` in the query function context (`queryFn: ({ signal }) => ...`). Thread it into `fetch` only if the networking helper accepts it. Even without it, a response for an obsolete key is stored under that key and never shown under the current one, so no manual stale check is needed.
11. After a mutation that changes server state, invalidate by prefix: `queryClient.invalidateQueries({ queryKey: thingKeys.all })`.

## Testing

React Query retries failures three times with backoff, so tests that exercise errors need `retry: false`, and a cache shared between tests leaks results across them.

Prefer a fresh client per test:

```tsx
const createWrapper = () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

renderHook(() => useThing(request), { wrapper: createWrapper() });
```

When a file already uses `renderWithProviders` from `tests/test-utils.tsx`, its shared `testQueryClient` has `staleTime: Infinity` and `refetchOnMount: false`, so call `testQueryClient.clear()` in `beforeEach`, or a later test will read the earlier test's cached data and never call the mocked network.

Mock the networking function (`vi.mock("@/components/networking", ...)`), not the query hook, so the test exercises the real key and enabled logic. Assert on what the user sees (loading text, error text, rows) and on which network calls were made with which arguments. Use `findBy*`/`waitFor` for anything that resolves after a fetch. An out-of-order test is still useful after migrating: resolve the old key's promise after the new key's and assert the old data never renders.

Run only the affected test files (`npx vitest run <paths>`), never the full suite without paths.
