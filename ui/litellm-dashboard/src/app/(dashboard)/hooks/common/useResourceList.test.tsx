import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ColumnFiltersState } from "@tanstack/react-table";
import { act, renderHook, waitFor } from "@testing-library/react";
import { NuqsTestingAdapter, type OnUrlUpdateFunction, type UrlUpdateEvent } from "nuqs/adapters/testing";
import React, { type PropsWithChildren } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  joinArrayFilters,
  splitArrayFilters,
  toSortParam,
  useResourceList,
  type ResourceListPage,
  type ResourceListQuery,
  type ResourceListUrlState,
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

const URL_STATE: ResourceListUrlState = {
  sortFields: ["created_at", "max_budget"],
  filterColumns: ["colour"],
};

const calls: ResourceListQuery[] = [];

interface UrlOptions {
  searchParams?: string;
  onUrlUpdate?: OnUrlUpdateFunction;
}

const renderList = (overrides: Partial<UseResourceListOptions<Row>> = {}, url: UrlOptions = {}) => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  const wrapper = ({ children }: PropsWithChildren) => (
    <NuqsTestingAdapter
      searchParams={url.searchParams}
      onUrlUpdate={url.onUrlUpdate}
      hasMemory
      resetUrlUpdateQueueOnMount={false}
    >
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    </NuqsTestingAdapter>
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
    urlState: URL_STATE,
    ...overrides,
  };
  return renderHook(() => useResourceList<Row>(options), { wrapper });
};

const lastCall = (): ResourceListQuery => calls[calls.length - 1];

const lastUrl = (onUrlUpdate: ReturnType<typeof vi.fn<OnUrlUpdateFunction>>): URLSearchParams => {
  const events = onUrlUpdate.mock.calls.map(([event]: [UrlUpdateEvent]) => event);
  return events[events.length - 1].searchParams;
};

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

describe("splitArrayFilters", () => {
  it("turns a comma separated URL value into an array for the listed columns only", () => {
    const filters: ColumnFiltersState = [
      { id: "providers", value: "openai, anthropic,,vertex_ai" },
      { id: "owner", value: "a,b" },
    ];

    expect(splitArrayFilters(filters, ["providers"])).toEqual([
      { id: "providers", value: ["openai", "anthropic", "vertex_ai"] },
      { id: "owner", value: "a,b" },
    ]);
  });
});

describe("joinArrayFilters", () => {
  it("joins the listed array columns into one URL string and leaves other filters alone", () => {
    const filters: ColumnFiltersState = [
      { id: "providers", value: ["openai", "anthropic"] },
      { id: "mode", value: [] },
      { id: "owner", value: "alice" },
    ];

    expect(joinArrayFilters(filters, ["providers", "mode"])).toEqual([
      { id: "providers", value: "openai,anthropic" },
      { id: "mode", value: "" },
      { id: "owner", value: "alice" },
    ]);
  });

  it("round-trips through splitArrayFilters", () => {
    const filters: ColumnFiltersState = [{ id: "features", value: ["vision", "function_calling"] }];

    expect(splitArrayFilters(joinArrayFilters(filters, ["features"]), ["features"])).toEqual(filters);
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

  it("reports loading while a new search request is still pending", async () => {
    let resolveSecond: ((value: ResourceListPage<Row>) => void) | undefined;
    const fetchPage = vi.fn((query: ResourceListQuery) => {
      calls.push(query);
      if (calls.length === 1) return Promise.resolve(page([{ id: "a" }], 3));
      return new Promise<ResourceListPage<Row>>((resolve) => {
        resolveSecond = resolve;
      });
    });
    const { result } = renderList({ fetchPage });
    await waitFor(() => expect(result.current.rows).toEqual([{ id: "a" }]));
    expect(result.current.isLoading).toBe(false);

    act(() => result.current.onSearchChange("zzz"));
    await waitFor(() => expect(lastCall().q).toBe("zzz"));
    expect(result.current.isLoading).toBe(true);

    act(() => resolveSecond?.(page([], 0)));
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.rows).toEqual([]);
  });

  it("surfaces a failed page as an error instead of empty rows", async () => {
    const fetchPage = vi.fn(() => Promise.reject(new Error("boom")));
    const { result } = renderList({ fetchPage });
    await waitFor(() => expect(result.current.error?.message).toBe("boom"));
    expect(result.current.rows).toEqual([]);
  });

  describe("URL state", () => {
    it("restores page, page size, sort and search from the URL into the very first request", async () => {
      const { result } = renderList(
        {},
        { searchParams: "?page=3&page_size=25&sort_by=max_budget&sort_order=asc&q=team-a" },
      );

      const expectedQuery: ResourceListQuery = { page: 3, page_size: 25, sort: "max_budget", q: "team-a" };
      await waitFor(() => expect(calls).toHaveLength(1));
      expect(calls[0]).toEqual(expectedQuery);
      expect(result.current.pagination).toEqual({ pageIndex: 2, pageSize: 25 });
      expect(result.current.sorting).toEqual([{ id: "max_budget", desc: false }]);
      expect(result.current.searchValue).toBe("team-a");
    });

    it("falls back to the default sort field when the URL names one the route cannot sort on", async () => {
      renderList({}, { searchParams: "?sort_by=api_key&sort_order=desc" });

      await waitFor(() => expect(calls).toHaveLength(1));
      expect(calls[0].sort).toBe("-created_at");
    });

    it("writes sort, page, page size and search changes to the URL", async () => {
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      const { result } = renderList({}, { onUrlUpdate });
      await waitFor(() => expect(calls).toHaveLength(1));

      act(() => result.current.onPaginationChange({ pageIndex: 2, pageSize: 25 }));
      await waitFor(() => expect(lastUrl(onUrlUpdate).get("page")).toBe("3"));
      expect(lastUrl(onUrlUpdate).get("page_size")).toBe("25");

      act(() => result.current.onSortingChange([{ id: "max_budget", desc: false }]));
      await waitFor(() => expect(lastUrl(onUrlUpdate).get("sort_by")).toBe("max_budget"));
      expect(lastUrl(onUrlUpdate).get("sort_order")).toBe("asc");
      expect(lastUrl(onUrlUpdate).has("page")).toBe(false);

      act(() => result.current.onSearchChange("team-b"));
      await waitFor(() => expect(lastUrl(onUrlUpdate).get("q")).toBe("team-b"));
      expect(lastUrl(onUrlUpdate).has("search")).toBe(false);
    });

    it("prefixes every key when the list shares its URL with another table", async () => {
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      const urlState: ResourceListUrlState = { ...URL_STATE, keyPrefix: "archived_" };
      const { result } = renderList({ urlState }, { searchParams: "?archived_page=2&page=7", onUrlUpdate });

      await waitFor(() => expect(calls).toHaveLength(1));
      expect(calls[0].page).toBe(2);

      act(() => result.current.onSearchChange("old"));
      await waitFor(() => expect(lastUrl(onUrlUpdate).get("archived_q")).toBe("old"));
      expect(lastUrl(onUrlUpdate).get("page")).toBe("7");
    });

    it("hands array filters from the URL to the serializer as arrays", async () => {
      const serializeFilters = vi.fn((filters: ColumnFiltersState) =>
        Object.fromEntries(filters.map((filter) => [filter.id, JSON.stringify(filter.value)])),
      );
      const urlState: ResourceListUrlState = { ...URL_STATE, arrayFilterColumns: ["colour"] };
      const { result } = renderList({ serializeFilters, urlState }, { searchParams: "?filter_colour=red,blue" });

      await waitFor(() => expect(calls).toHaveLength(1));
      expect(result.current.columnFilters).toEqual([{ id: "colour", value: ["red", "blue"] }]);
      expect(calls[0].colour).toBe('["red","blue"]');
    });

    it("stores array filters comma separated and drops the key once the array is empty", async () => {
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      const urlState: ResourceListUrlState = { ...URL_STATE, arrayFilterColumns: ["colour"] };
      const { result } = renderList({ urlState }, { searchParams: "?page=4", onUrlUpdate });
      await waitFor(() => expect(calls).toHaveLength(1));

      act(() => result.current.onColumnFiltersChange([{ id: "colour", value: ["green", "teal"] }]));
      await waitFor(() => expect(lastUrl(onUrlUpdate).get("filter_colour")).toBe("green,teal"));
      expect(lastUrl(onUrlUpdate).has("page")).toBe(false);
      await waitFor(() => expect(result.current.columnFilters).toEqual([{ id: "colour", value: ["green", "teal"] }]));

      const withoutColour = (previous: ColumnFiltersState) => previous.filter((filter) => filter.id !== "colour");
      act(() => result.current.onColumnFiltersChange(withoutColour));
      await waitFor(() => expect(lastUrl(onUrlUpdate).has("filter_colour")).toBe(false));
    });

    it("runs filters through the consumer's codec in both directions", async () => {
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      const urlState: ResourceListUrlState = {
        sortFields: ["created_at"],
        filterColumns: ["size_min"],
        toUrlFilters: (filters) => [
          { id: "size_min", value: (filters.find((f) => f.id === "size")?.value as { min?: string })?.min ?? "" },
        ],
        fromUrlFilters: (filters) =>
          filters.map((filter) => (filter.id === "size_min" ? { id: "size", value: { min: filter.value } } : filter)),
      };
      const { result } = renderList({ urlState }, { searchParams: "?filter_size_min=5", onUrlUpdate });

      await waitFor(() => expect(result.current.columnFilters).toEqual([{ id: "size", value: { min: "5" } }]));

      act(() => result.current.onColumnFiltersChange([{ id: "size", value: { min: "9" } }]));
      await waitFor(() => expect(lastUrl(onUrlUpdate).get("filter_size_min")).toBe("9"));
    });
  });
});
