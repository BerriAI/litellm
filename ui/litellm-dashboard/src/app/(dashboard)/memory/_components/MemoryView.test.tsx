import type { PaginationState } from "@tanstack/react-table";
import { act, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders, testQueryClient } from "@/../tests/test-utils";
import { MemoryRow } from "@/components/networking";

import { MemoryView } from "./MemoryView";

interface CapturedTableProps {
  isLoading: boolean;
  rowCount: number;
  data: MemoryRow[];
  pagination: PaginationState;
  searchValue: string;
  hasActiveSearch: boolean;
  onSearchChange: (value: string) => void;
  onPaginationChange: (state: PaginationState) => void;
  onViewClick: (row: MemoryRow) => void;
}

const captured = vi.hoisted(() => ({ current: null as CapturedTableProps | null }));
const fetchMemoryListMock = vi.hoisted(() => vi.fn());

vi.mock("./MemoryTable", () => ({
  MemoryTable: function MemoryTableMock(props: CapturedTableProps) {
    captured.current = props;
    return <div data-testid="memory-table-mock" />;
  },
}));

vi.mock("@/components/networking", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/components/networking")>()),
  fetchMemoryList: fetchMemoryListMock,
}));

vi.mock("@tanstack/react-pacer/debouncer", () => ({
  useDebouncedValue: (value: unknown) => [value, { cancel: vi.fn(), flush: vi.fn() }],
}));

interface RenderViewOptions {
  searchParams?: Record<string, string>;
  onUrlUpdate?: OnUrlUpdateFunction;
}

const renderView = (accessToken: string | null, options: RenderViewOptions = {}) =>
  renderWithProviders(<MemoryView accessToken={accessToken} userID={null} userRole={null} />, options);

const drawerRow: MemoryRow = {
  memory_id: "mem-drawer",
  key: "user:profile",
  value: "remembered",
  metadata: null,
  user_id: null,
  team_id: null,
};

const lastUrl = (onUrlUpdate: ReturnType<typeof vi.fn<OnUrlUpdateFunction>>) => onUrlUpdate.mock.calls.at(-1)?.[0];

describe("MemoryView", () => {
  beforeEach(() => {
    testQueryClient.clear();
    fetchMemoryListMock.mockReset();
    fetchMemoryListMock.mockResolvedValue({ memories: [], total: 0 });
  });

  it("queries the server with the search box value as `search` and resets to page 1", async () => {
    renderView("token", { searchParams: { page: "3" } });
    await waitFor(() =>
      expect(fetchMemoryListMock).toHaveBeenLastCalledWith("token", expect.objectContaining({ page: 3 })),
    );

    act(() => captured.current?.onSearchChange("mem-abc123"));

    await waitFor(() =>
      expect(fetchMemoryListMock).toHaveBeenLastCalledWith("token", { search: "mem-abc123", page: 1, pageSize: 50 }),
    );
    expect(captured.current?.hasActiveSearch).toBe(true);
  });

  it("keeps the table out of the skeleton state when the token is null (disabled query)", () => {
    renderView(null);

    expect(captured.current).not.toBeNull();
    expect(captured.current?.isLoading).toBe(false);
    expect(captured.current?.data).toEqual([]);
    expect(captured.current?.rowCount).toBe(0);
    expect(captured.current?.hasActiveSearch).toBe(false);
  });

  it("heads the page with the Memory title and the /v1/memory scope note", () => {
    renderView(null);

    expect(screen.getByRole("heading", { name: "Memory" })).toBeInTheDocument();
    expect(screen.getByText("/v1/memory")).toBeInTheDocument();
    expect(screen.getByText(/Scoped to memories visible to your user \/ team \(admins see all\)/)).toBeInTheDocument();
  });

  it("opens the create modal from the New memory button", async () => {
    const user = userEvent.setup();
    renderView(null);

    expect(screen.queryByText("Create memory")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /new memory/i }));

    expect(await screen.findByText("Create memory")).toBeInTheDocument();
  });

  it("opens the detail drawer for the row the table hands back, and closes it again", async () => {
    fetchMemoryListMock.mockResolvedValue({ memories: [drawerRow], total: 1 });
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderView("token", { onUrlUpdate });
    await waitFor(() => expect(captured.current?.data).toEqual([drawerRow]));

    expect(screen.queryByText("Memory ID")).not.toBeInTheDocument();

    act(() => captured.current?.onViewClick(drawerRow));

    expect(await screen.findByText("Memory ID")).toBeInTheDocument();
    expect(screen.getByText("mem-drawer")).toBeInTheDocument();
    expect(lastUrl(onUrlUpdate)?.searchParams.get("memory")).toBe("mem-drawer");
    expect(lastUrl(onUrlUpdate)?.options.history).toBe("push");

    await user.click(screen.getByRole("button", { name: /close/i }));

    await waitFor(() => expect(screen.queryByText("mem-drawer")).not.toBeInTheDocument());
    expect(lastUrl(onUrlUpdate)?.searchParams.has("memory")).toBe(false);
  });

  describe("URL state", () => {
    it("queries the server with the search, page and page_size from the URL", async () => {
      renderView("token", { searchParams: { search: "abc", page: "3", page_size: "25" } });

      await waitFor(() =>
        expect(fetchMemoryListMock).toHaveBeenLastCalledWith("token", { search: "abc", page: 3, pageSize: 25 }),
      );
      expect(captured.current?.searchValue).toBe("abc");
      expect(captured.current?.pagination).toEqual({ pageIndex: 2, pageSize: 25 });
    });

    it("defaults to 50 rows per page without page_size in the URL", async () => {
      renderView("token");

      await waitFor(() =>
        expect(fetchMemoryListMock).toHaveBeenLastCalledWith("token", { search: undefined, page: 1, pageSize: 50 }),
      );
    });

    it("writes the search to the URL and drops the page", async () => {
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderView("token", { searchParams: { page: "3" }, onUrlUpdate });

      act(() => captured.current?.onSearchChange("abc"));

      await waitFor(() => expect(lastUrl(onUrlUpdate)?.searchParams.get("search")).toBe("abc"));
      expect(lastUrl(onUrlUpdate)?.searchParams.has("page")).toBe(false);
    });

    it("writes the page and page size to the URL", async () => {
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderView("token", { onUrlUpdate });

      act(() => captured.current?.onPaginationChange({ pageIndex: 1, pageSize: 25 }));

      await waitFor(() => expect(lastUrl(onUrlUpdate)?.searchParams.get("page")).toBe("2"));
      expect(lastUrl(onUrlUpdate)?.searchParams.get("page_size")).toBe("25");
    });

    it("opens the drawer for the memory in the URL once the page has loaded it", async () => {
      fetchMemoryListMock.mockResolvedValue({ memories: [drawerRow], total: 1 });
      renderView("token", { searchParams: { memory: "mem-drawer" } });

      expect(await screen.findByText("Memory ID")).toBeInTheDocument();
      expect(screen.getByText("mem-drawer")).toBeInTheDocument();
    });

    it("keeps the drawer closed when the memory in the URL is not on the loaded page", async () => {
      fetchMemoryListMock.mockResolvedValue({ memories: [drawerRow], total: 1 });
      renderView("token", { searchParams: { memory: "mem-elsewhere" } });

      await waitFor(() => expect(captured.current?.data).toEqual([drawerRow]));
      expect(screen.queryByText("Memory ID")).not.toBeInTheDocument();
    });
  });
});
