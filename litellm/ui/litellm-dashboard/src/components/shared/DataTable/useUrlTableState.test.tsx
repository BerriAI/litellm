import { SortingState } from "@tanstack/react-table";
import { act, renderHook, waitFor } from "@testing-library/react";
import { withNuqsTestingAdapter, type OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { describe, expect, it, Mock, vi } from "vitest";
import { useUrlTableState, type UrlTableStateOptions } from "./useUrlTableState";

const FILTER_COLUMNS = ["team_id", "user_id"] as const;
type FilterColumn = (typeof FILTER_COLUMNS)[number];

const BASE_OPTIONS: UrlTableStateOptions<FilterColumn> = {
  sortFields: ["created_at", "spend", "key_alias"],
  defaultSort: { id: "created_at", desc: true },
  defaultPageSize: 50,
  filterColumns: FILTER_COLUMNS,
};

const PREFIXED_AND_UNPREFIXED_PARAMS = {
  audit_page: "2",
  audit_page_size: "10",
  audit_search: "prefixed",
  audit_sort_by: "spend",
  audit_sort_order: "asc",
  audit_filter_team_id: "team-1",
  page: "5",
  search: "unprefixed",
  filter_team_id: "other-team",
};

const RENAMED_AND_DEFAULT_PARAMS = {
  key_search: "prod",
  filter_team: "team-1",
  search: "ignored",
  filter_team_id: "ignored",
};

const flipDirection = (previous: SortingState): SortingState => previous.map((sort) => ({ ...sort, desc: !sort.desc }));

const renderTableState = (
  searchParams: Record<string, string> = {},
  overrides: Partial<UrlTableStateOptions<FilterColumn>> = {},
) => {
  const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
  const options = { ...BASE_OPTIONS, ...overrides };
  const hook = renderHook(() => useUrlTableState(options), {
    wrapper: withNuqsTestingAdapter({ searchParams, onUrlUpdate, hasMemory: true }),
  });
  return { ...hook, onUrlUpdate };
};

const lastUrl = (onUrlUpdate: Mock<OnUrlUpdateFunction>) => {
  const event = onUrlUpdate.mock.calls.at(-1)?.[0];
  if (!event) throw new Error("no URL update was emitted");
  return event;
};

const flushUrl = async (onUrlUpdate: Mock<OnUrlUpdateFunction>, write: () => void) => {
  const callsBefore = onUrlUpdate.mock.calls.length;
  await act(async () => {
    write();
  });
  await waitFor(() => expect(onUrlUpdate.mock.calls.length).toBeGreaterThan(callsBefore));
  return lastUrl(onUrlUpdate).searchParams;
};

describe("reading table state from the URL", () => {
  it("falls back to the defaults when the URL carries no table state", () => {
    const { result } = renderTableState();

    expect(result.current.search).toBe("");
    expect(result.current.sorting).toEqual([{ id: "created_at", desc: true }]);
    expect(result.current.pagination).toEqual({ pageIndex: 0, pageSize: 50 });
    expect(result.current.columnFilters).toEqual([]);
  });

  it("maps the 1-based page and page_size onto TanStack pagination", () => {
    const { result } = renderTableState({ page: "3", page_size: "25" });

    expect(result.current.pagination).toEqual({ pageIndex: 2, pageSize: 25 });
  });

  it.each(["0", "-3", "not-a-number"])("clamps a page of %s up to the first page", (page) => {
    const { result } = renderTableState({ page });

    expect(result.current.pagination.pageIndex).toBe(0);
  });

  it.each([
    ["1000", undefined, 100],
    ["1000", 20, 20],
    ["0", undefined, 1],
  ])("clamps a page_size of %s with maxPageSize %s to %s", (pageSize, maxPageSize, expected) => {
    const { result } = renderTableState({ page_size: pageSize }, { maxPageSize });

    expect(result.current.pagination.pageSize).toBe(expected);
  });

  it("reads a sortable sort_by and its sort_order", () => {
    const { result } = renderTableState({ sort_by: "spend", sort_order: "asc" });

    expect(result.current.sorting).toEqual([{ id: "spend", desc: false }]);
  });

  it("resolves a sort_by outside the allow-list to the default column while keeping the URL's direction", () => {
    const { result } = renderTableState({ sort_by: "totally_unknown", sort_order: "asc" });

    expect(result.current.sorting).toEqual([{ id: "created_at", desc: false }]);
  });

  it("maps filter_<column> params onto columnFilters, trimming whitespace and dropping blanks", () => {
    const { result } = renderTableState({ filter_team_id: "team-1", filter_user_id: "   " });

    expect(result.current.columnFilters).toEqual([{ id: "team_id", value: "team-1" }]);

    const trimmed = renderTableState({ filter_user_id: "  user-42  " });
    expect(trimmed.result.current.columnFilters).toEqual([{ id: "user_id", value: "user-42" }]);
  });

  it("reads the search term verbatim so the input can hold trailing spaces", () => {
    const { result } = renderTableState({ search: "prod " });

    expect(result.current.search).toBe("prod ");
  });

  it("reads every key under keyPrefix and ignores the unprefixed ones", () => {
    const { result } = renderTableState(PREFIXED_AND_UNPREFIXED_PARAMS, { keyPrefix: "audit_" });

    expect(result.current.pagination).toEqual({ pageIndex: 1, pageSize: 10 });
    expect(result.current.search).toBe("prefixed");
    expect(result.current.sorting).toEqual([{ id: "spend", desc: false }]);
    expect(result.current.columnFilters).toEqual([{ id: "team_id", value: "team-1" }]);
  });

  it("reads renamed keys from urlKeys and ignores the default names", () => {
    const { result } = renderTableState(RENAMED_AND_DEFAULT_PARAMS, {
      urlKeys: { search: "key_search", filter_team_id: "filter_team" },
    });

    expect(result.current.search).toBe("prod");
    expect(result.current.columnFilters).toEqual([{ id: "team_id", value: "team-1" }]);
  });

  it("applies keyPrefix in front of a renamed key", () => {
    const { result } = renderTableState(
      { audit_key_search: "prod", key_search: "ignored" },
      { keyPrefix: "audit_", urlKeys: { search: "key_search" } },
    );

    expect(result.current.search).toBe("prod");
  });
});

describe("writing table state to the URL", () => {
  it("resolves a function updater against the current pagination and replaces history", async () => {
    const { result, onUrlUpdate } = renderTableState({ page: "2" });

    const url = await flushUrl(onUrlUpdate, () =>
      result.current.onPaginationChange((previous) => ({ ...previous, pageIndex: previous.pageIndex + 1 })),
    );

    expect(url.get("page")).toBe("3");
    expect(url.has("page_size")).toBe(false);
    expect(lastUrl(onUrlUpdate).options.history).toBe("replace");
    expect(result.current.pagination).toEqual({ pageIndex: 2, pageSize: 50 });
  });

  it("writes page_size and drops it again once it returns to the default", async () => {
    const { result, onUrlUpdate } = renderTableState();

    const withSize = await flushUrl(onUrlUpdate, () =>
      result.current.onPaginationChange({ pageIndex: 0, pageSize: 25 }),
    );
    expect(withSize.get("page_size")).toBe("25");
    expect(withSize.has("page")).toBe(false);

    const backToDefault = await flushUrl(onUrlUpdate, () =>
      result.current.onPaginationChange({ pageIndex: 0, pageSize: 50 }),
    );
    expect(backToDefault.has("page_size")).toBe(false);
  });

  it("setSearch writes the term and returns to the first page", async () => {
    const { result, onUrlUpdate } = renderTableState({ page: "3" });

    const url = await flushUrl(onUrlUpdate, () => result.current.setSearch("prod"));

    expect(url.get("search")).toBe("prod");
    expect(url.has("page")).toBe(false);
    expect(result.current.search).toBe("prod");
    expect(result.current.pagination.pageIndex).toBe(0);
  });

  it("setSearch with an empty string removes the key", async () => {
    const { result, onUrlUpdate } = renderTableState({ search: "prod" });

    const url = await flushUrl(onUrlUpdate, () => result.current.setSearch(""));

    expect(url.has("search")).toBe(false);
    expect(result.current.search).toBe("");
  });

  it("onSortingChange writes sort_by and sort_order and returns to the first page", async () => {
    const { result, onUrlUpdate } = renderTableState({ page: "3" });

    const url = await flushUrl(onUrlUpdate, () => result.current.onSortingChange([{ id: "spend", desc: false }]));

    expect(url.get("sort_by")).toBe("spend");
    expect(url.get("sort_order")).toBe("asc");
    expect(url.has("page")).toBe(false);
    expect(result.current.sorting).toEqual([{ id: "spend", desc: false }]);
  });

  it("onSortingChange drops the keys when the sort matches the default or is cleared", async () => {
    const { result, onUrlUpdate } = renderTableState({ sort_by: "spend", sort_order: "asc" });

    const explicitDefault = await flushUrl(onUrlUpdate, () =>
      result.current.onSortingChange([{ id: "created_at", desc: true }]),
    );
    expect(explicitDefault.has("sort_by")).toBe(false);
    expect(explicitDefault.has("sort_order")).toBe(false);

    await flushUrl(onUrlUpdate, () => result.current.onSortingChange([{ id: "key_alias", desc: false }]));
    const cleared = await flushUrl(onUrlUpdate, () => result.current.onSortingChange([]));
    expect(cleared.has("sort_by")).toBe(false);
    expect(cleared.has("sort_order")).toBe(false);
    expect(result.current.sorting).toEqual([{ id: "created_at", desc: true }]);
  });

  it("onSortingChange resolves a function updater against the current sort", async () => {
    const { result, onUrlUpdate } = renderTableState({ sort_by: "spend" });

    const url = await flushUrl(onUrlUpdate, () => result.current.onSortingChange(flipDirection));

    expect(url.get("sort_by")).toBe("spend");
    expect(url.get("sort_order")).toBe("asc");
    expect(result.current.sorting).toEqual([{ id: "spend", desc: false }]);
  });

  it("onColumnFiltersChange writes trimmed filter_<column> keys and returns to the first page", async () => {
    const { result, onUrlUpdate } = renderTableState({ page: "3" });

    const url = await flushUrl(onUrlUpdate, () =>
      result.current.onColumnFiltersChange([{ id: "team_id", value: "  team-1  " }]),
    );

    expect(url.get("filter_team_id")).toBe("team-1");
    expect(url.has("page")).toBe(false);
    expect(result.current.columnFilters).toEqual([{ id: "team_id", value: "team-1" }]);
  });

  it("onColumnFiltersChange removes the key for an empty value and for a filter no longer present", async () => {
    const { result, onUrlUpdate } = renderTableState({ filter_team_id: "team-1", filter_user_id: "user-42" });

    const url = await flushUrl(onUrlUpdate, () => result.current.onColumnFiltersChange([{ id: "team_id", value: "" }]));

    expect(url.has("filter_team_id")).toBe(false);
    expect(url.has("filter_user_id")).toBe(false);
    expect(result.current.columnFilters).toEqual([]);
  });

  it("onColumnFiltersChange ignores a non-string filter value", async () => {
    const { result, onUrlUpdate } = renderTableState({ filter_team_id: "team-1" });

    const url = await flushUrl(onUrlUpdate, () =>
      result.current.onColumnFiltersChange([{ id: "team_id", value: ["team-1", "team-2"] }]),
    );

    expect(url.has("filter_team_id")).toBe(false);
  });

  it("onColumnFiltersChange resolves a function updater against the current filters", async () => {
    const { result, onUrlUpdate } = renderTableState({ filter_team_id: "team-1" });

    const url = await flushUrl(onUrlUpdate, () =>
      result.current.onColumnFiltersChange((previous) => [...previous, { id: "user_id", value: "user-42" }]),
    );

    expect(url.get("filter_team_id")).toBe("team-1");
    expect(url.get("filter_user_id")).toBe("user-42");
  });

  it("writes prefixed and renamed keys only", async () => {
    const { result, onUrlUpdate } = renderTableState(
      {},
      { keyPrefix: "audit_", urlKeys: { search: "key_search", filter_team_id: "filter_team" } },
    );

    await flushUrl(onUrlUpdate, () => result.current.setSearch("prod"));
    await flushUrl(onUrlUpdate, () => result.current.onSortingChange([{ id: "spend", desc: false }]));
    const url = await flushUrl(onUrlUpdate, () =>
      result.current.onColumnFiltersChange([{ id: "team_id", value: "team-1" }]),
    );

    expect(url.get("audit_key_search")).toBe("prod");
    expect(url.get("audit_sort_by")).toBe("spend");
    expect(url.get("audit_filter_team")).toBe("team-1");
    expect([...url.keys()].filter((key) => !key.startsWith("audit_"))).toEqual([]);
    expect(url.has("audit_search")).toBe(false);
    expect(url.has("audit_filter_team_id")).toBe(false);
  });
});

describe("referential stability", () => {
  it("keeps the TanStack state and the page-clamp handler stable across rerenders while the URL is unchanged", () => {
    const { result, rerender } = renderTableState({ page: "2", filter_team_id: "team-1", sort_by: "spend" });
    const first = result.current;

    rerender();

    expect(result.current.sorting).toBe(first.sorting);
    expect(result.current.pagination).toBe(first.pagination);
    expect(result.current.columnFilters).toBe(first.columnFilters);
    expect(result.current.onPaginationChange).toBe(first.onPaginationChange);
  });

  it("hands out new pagination and untouched sorting after a page change", async () => {
    const { result, onUrlUpdate } = renderTableState({ sort_by: "spend" });
    const first = result.current;

    await flushUrl(onUrlUpdate, () => result.current.onPaginationChange({ pageIndex: 4, pageSize: 50 }));

    expect(result.current.pagination).not.toBe(first.pagination);
    expect(result.current.pagination.pageIndex).toBe(4);
    expect(result.current.sorting).toBe(first.sorting);
  });
});
