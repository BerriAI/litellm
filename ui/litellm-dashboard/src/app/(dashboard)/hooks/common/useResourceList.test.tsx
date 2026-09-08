import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ColumnFiltersState } from "@tanstack/react-table";
import { act, renderHook, waitFor } from "@testing-library/react";
import React, { type PropsWithChildren } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  toSortParam,
  useResourceList,
  type ResourceListPage,
  type ResourceListQuery,
  type UseResourceListOptions,
} from "./useResourceList";

interface Row {
  id: string;
}

const page = (rows: Row[], totalCount: number): ResourceListPage<Row> => ({
  data: rows,
  meta: { total_count: totalCount, page: 1, page_size: 50, total_pages: 1 },
});

const noFilters = (): Readonly<Record<string, string>> => ({});

const calls: ResourceListQuery[] = [];

const renderList = (overrides: Partial<UseResourceListOptions<Row>> = {}) => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  const wrapper = ({ children }: PropsWithChildren) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
  const fetchPage = vi.fn((query: ResourceListQuery) => {
    calls.push(query);
    return Promise.resolve(page([{ id: "a" }], 3));
  });
  const options: UseResourceListOptions<Row> = {
    queryKey: ["widgets", "list"],
    fetchPage,
    serializeFilters: noFilters,
    defaultSorting: [{ id: "created_at", desc: true }],
    defaultPageSize: 50,
    enabled: true,
    ...overrides,
  };
  return renderHook(() => useResourceList<Row>(options), { wrapper });
};

const lastCall = (): ResourceListQuery => calls[calls.length - 1];

describe("toSortParam", () => {
  it("prefixes descending fields with a minus and joins with commas", () => {
    expect(toSortParam([{ id: "created_at", desc: true }])).toBe("-created_at");
    expect(toSortParam([{ id: "max_budget", desc: false }])).toBe("max_budget");
    expect(
      toSortParam([
        { id: "a", desc: false },
        { id: "b", desc: true },
      ]),
    ).toBe("a,-b");
  });
});

describe("useResourceList", () => {
  beforeEach(() => {
    calls.length = 0;
  });

  it("requests the first page with the default sort", async () => {
    const { result } = renderList();
    await waitFor(() => expect(result.current.rowCount).toBe(3));
    expect(lastCall()).toEqual({ page: 1, page_size: 50, sort: "-created_at" });
  });

  it("exposes the returned rows and total count", async () => {
    const { result } = renderList();
    await waitFor(() => expect(result.current.rows).toEqual([{ id: "a" }]));
    expect(result.current.rowCount).toBe(3);
  });

  it("does not fetch while disabled", async () => {
    const { result } = renderList({ enabled: false });
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(calls).toHaveLength(0);
  });

  it("sends the new sort and returns to the first page", async () => {
    const { result } = renderList();
    await waitFor(() => expect(calls).toHaveLength(1));

    act(() => result.current.onPaginationChange({ pageIndex: 2, pageSize: 50 }));
    await waitFor(() => expect(lastCall().page).toBe(3));

    act(() => result.current.onSortingChange([{ id: "max_budget", desc: false }]));
    await waitFor(() => expect(lastCall().sort).toBe("max_budget"));
    expect(lastCall().page).toBe(1);
  });

  it("omits sort entirely when nothing is sorted", async () => {
    const { result } = renderList({ defaultSorting: [] });
    await waitFor(() => expect(calls).toHaveLength(1));
    expect(result.current.sorting).toEqual([]);
    expect(lastCall()).not.toHaveProperty("sort");
  });

  it("debounces the search into a single trimmed q and returns to the first page", async () => {
    const { result } = renderList();
    await waitFor(() => expect(calls).toHaveLength(1));

    act(() => result.current.onPaginationChange({ pageIndex: 1, pageSize: 50 }));
    await waitFor(() => expect(lastCall().page).toBe(2));

    act(() => result.current.onSearchChange("bud"));
    act(() => result.current.onSearchChange("budg "));

    await waitFor(() => expect(lastCall().q).toBe("budg"));
    expect(lastCall().page).toBe(1);
    expect(calls.some((call) => call.q === "bud")).toBe(false);
  });

  it("stops sending q once the search box is cleared", async () => {
    const { result } = renderList();
    act(() => result.current.onSearchChange("budget"));
    await waitFor(() => expect(lastCall().q).toBe("budget"));

    act(() => result.current.onSearchChange(""));
    await waitFor(() => expect(lastCall()).not.toHaveProperty("q"));
  });

  it("merges serialized filters into the request and returns to the first page", async () => {
    const serializeFilters = (filters: ColumnFiltersState): Readonly<Record<string, string>> =>
      filters.length === 0 ? {} : { "filter[colour][in]": String(filters[0].value) };
    const { result } = renderList({ serializeFilters });
    await waitFor(() => expect(calls).toHaveLength(1));

    act(() => result.current.onPaginationChange({ pageIndex: 3, pageSize: 50 }));
    await waitFor(() => expect(lastCall().page).toBe(4));

    act(() => result.current.onColumnFiltersChange([{ id: "colour", value: "red" }]));
    await waitFor(() => expect(lastCall()["filter[colour][in]"]).toBe("red"));
    expect(lastCall().page).toBe(1);

    act(() => result.current.onColumnFiltersChange([]));
    await waitFor(() => expect(lastCall()).not.toHaveProperty("filter[colour][in]"));
  });

  it("sends the requested page size", async () => {
    const { result } = renderList();
    await waitFor(() => expect(calls).toHaveLength(1));

    act(() => result.current.onPaginationChange({ pageIndex: 0, pageSize: 25 }));
    await waitFor(() => expect(lastCall().page_size).toBe(25));
  });

  it("surfaces a failed page as an error instead of empty rows", async () => {
    const fetchPage = vi.fn(() => Promise.reject(new Error("boom")));
    const { result } = renderList({ fetchPage });
    await waitFor(() => expect(result.current.error?.message).toBe("boom"));
    expect(result.current.rows).toEqual([]);
  });
});
