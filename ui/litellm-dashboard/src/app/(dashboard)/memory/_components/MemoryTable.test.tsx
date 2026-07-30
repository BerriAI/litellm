import { PaginationState } from "@tanstack/react-table";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React, { useState } from "react";
import { describe, expect, it, vi } from "vitest";

import { MemoryRow } from "@/components/networking";

import { MemoryTable } from "./MemoryTable";

const makeMemory = (overrides: Partial<MemoryRow> = {}): MemoryRow => ({
  memory_id: "mem-1",
  key: "user:profile",
  value: "The user prefers concise answers.",
  metadata: null,
  user_id: "user-42",
  team_id: "team-7",
  updated_at: "2024-05-01T12:00:00Z",
  ...overrides,
});

const baseProps = {
  data: [makeMemory()],
  isLoading: false,
  rowCount: 1,
  pagination: { pageIndex: 0, pageSize: 50 } as PaginationState,
  onPaginationChange: vi.fn(),
  searchValue: "",
  onSearchChange: vi.fn(),
  isRefreshing: false,
  onRefresh: vi.fn(),
  hasActiveSearch: false,
  onViewClick: vi.fn(),
  onEditClick: vi.fn(),
  onDeleteClick: vi.fn(),
};

describe("MemoryTable", () => {
  it("renders every column header", () => {
    render(<MemoryTable {...baseProps} />);
    for (const header of ["ID", "Name", "Preview", "User ID", "Team ID", "Updated"]) {
      expect(screen.getByText(header)).toBeInTheDocument();
    }
  });

  it("opens the detail view when the ID identity cell is clicked", async () => {
    const user = userEvent.setup();
    const onViewClick = vi.fn();
    const row = makeMemory({ memory_id: "mem-click" });
    render(<MemoryTable {...baseProps} data={[row]} onViewClick={onViewClick} />);

    await user.click(screen.getByText("mem-click"));

    expect(onViewClick).toHaveBeenCalledTimes(1);
    expect(onViewClick).toHaveBeenCalledWith(row);
  });

  it("routes each overflow-menu action to its callback with the row", async () => {
    const user = userEvent.setup();
    const onViewClick = vi.fn();
    const onEditClick = vi.fn();
    const onDeleteClick = vi.fn();
    const row = makeMemory({ memory_id: "mem-9" });
    render(
      <MemoryTable
        {...baseProps}
        data={[row]}
        onViewClick={onViewClick}
        onEditClick={onEditClick}
        onDeleteClick={onDeleteClick}
      />,
    );

    await user.click(screen.getByTestId("memory-actions-mem-9"));
    await user.click(await screen.findByTestId("memory-action-edit"));
    expect(onEditClick).toHaveBeenCalledWith(row);
    expect(onViewClick).not.toHaveBeenCalled();
    expect(onDeleteClick).not.toHaveBeenCalled();

    await user.click(screen.getByTestId("memory-actions-mem-9"));
    await user.click(await screen.findByTestId("memory-action-delete"));
    expect(onDeleteClick).toHaveBeenCalledWith(row);

    await user.click(screen.getByTestId("memory-actions-mem-9"));
    await user.click(await screen.findByTestId("memory-action-view"));
    expect(onViewClick).toHaveBeenCalledWith(row);
  });

  it("shows the empty-only copy when there is no data and no active search", () => {
    render(<MemoryTable {...baseProps} data={[]} rowCount={0} hasActiveSearch={false} />);
    expect(screen.getByText("No memories stored yet")).toBeInTheDocument();
    expect(screen.queryByText("No matching memories")).not.toBeInTheDocument();
  });

  it("shows the filtered-empty copy when a search is active", () => {
    render(<MemoryTable {...baseProps} data={[]} rowCount={0} hasActiveSearch={true} />);
    expect(screen.getByText("No matching memories")).toBeInTheDocument();
    expect(screen.queryByText("No memories stored yet")).not.toBeInTheDocument();
  });

  it("renders loading skeleton rows instead of the empty state while loading", () => {
    render(<MemoryTable {...baseProps} data={[]} rowCount={0} isLoading={true} />);
    expect(screen.getAllByTestId("skeleton-row").length).toBeGreaterThan(0);
    expect(screen.queryByText("No memories stored yet")).not.toBeInTheDocument();
  });

  it("drives the pagination footer from the server rowCount, not the page's row length", () => {
    render(<MemoryTable {...baseProps} data={[makeMemory()]} rowCount={120} />);
    const range = screen.getByTestId("pagination-range");
    expect(range).toHaveTextContent("Showing 1-50 of 120");
    expect(screen.getByTestId("pagination-page")).toHaveTextContent("Page 1 of 3");
    expect(screen.getByTestId("pagination-next")).toBeEnabled();
  });

  it("advances the page through the server pagination handler", async () => {
    const user = userEvent.setup();
    const onPaginationChange = vi.fn();
    render(<MemoryTable {...baseProps} rowCount={120} onPaginationChange={onPaginationChange} />);

    await user.click(screen.getByTestId("pagination-next"));

    expect(onPaginationChange).toHaveBeenCalled();
  });

  it("forwards toolbar search input and refresh to their callbacks", async () => {
    const user = userEvent.setup();
    const onSearchChange = vi.fn();
    const onRefresh = vi.fn();
    render(<MemoryTable {...baseProps} onSearchChange={onSearchChange} onRefresh={onRefresh} />);

    await user.type(screen.getByTestId("datatable-search"), "u");
    expect(onSearchChange).toHaveBeenCalledWith("u");

    await user.click(screen.getByTestId("datatable-refresh"));
    expect(onRefresh).toHaveBeenCalledTimes(1);
  });

  it("keeps the page in range when the rows-per-page selector shrinks the page count", async () => {
    const user = userEvent.setup();
    const rowCount = 120;
    const seen: PaginationState[] = [];

    function Harness() {
      const [pagination, setPagination] = useState<PaginationState>({ pageIndex: 4, pageSize: 25 });
      seen.push(pagination);
      return (
        <MemoryTable {...baseProps} rowCount={rowCount} pagination={pagination} onPaginationChange={setPagination} />
      );
    }

    render(<Harness />);
    expect(screen.getByTestId("pagination-page")).toHaveTextContent("Page 5 of 5");

    await user.click(screen.getByTestId("pagination-page-size"));
    await user.click(await screen.findByRole("option", { name: "100" }));

    const final = seen[seen.length - 1];
    expect(final.pageSize).toBe(100);
    expect(final.pageIndex).toBeLessThanOrEqual(Math.ceil(rowCount / final.pageSize) - 1);
    expect(screen.getByTestId("pagination-page")).toHaveTextContent("Page 2 of 2");
  });

  it("renders secondary id and date cells for the row", () => {
    render(<MemoryTable {...baseProps} data={[makeMemory({ user_id: "user-42", team_id: "team-7" })]} />);
    const table = screen.getByRole("table");
    expect(within(table).getByText("user-42")).toBeInTheDocument();
    expect(within(table).getByText("team-7")).toBeInTheDocument();
    expect(within(table).getByText("user:profile")).toBeInTheDocument();
  });
});
