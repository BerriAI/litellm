import type { ColumnDef } from "@tanstack/react-table";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type * as React from "react";
import { describe, expect, it, vi } from "vitest";

import { DataTable } from "./DataTable";
import { DataTableToolbar } from "./DataTableToolbar";

interface Person {
  id: string;
  name: string;
}

const DATA: Person[] = [
  { id: "a", name: "Alice" },
  { id: "b", name: "Bob" },
];

const columns: ColumnDef<Person, unknown>[] = [
  {
    accessorKey: "name",
    header: "Name",
    meta: { title: "Name" },
    filterFn: (row, columnId, value) => row.getValue<string>(columnId) === value,
    cell: ({ row }) => <span data-testid="name-cell">{row.original.name}</span>,
  },
];

const names = (): (string | null)[] => screen.getAllByTestId("name-cell").map((el) => el.textContent);

function Harness({
  onOpenFilters,
  onRefresh,
  children,
}: {
  onOpenFilters?: () => void;
  onRefresh?: () => void;
  children?: React.ReactNode;
}) {
  return (
    <DataTable
      data={DATA}
      columns={columns}
      filterMode="client"
      defaultColumnFilters={[{ id: "name", value: "Alice" }]}
      toolbar={(table) => (
        <DataTableToolbar table={table} onOpenFilters={onOpenFilters} onRefresh={onRefresh}>
          {children}
        </DataTableToolbar>
      )}
    />
  );
}

describe("DataTableToolbar", () => {
  it("renders a chip for each active filter with its label and value", () => {
    render(<Harness />);
    expect(names()).toEqual(["Alice"]);
    const chip = screen.getByTestId("filter-chip-name");
    expect(chip).toHaveTextContent("Name:");
    expect(chip).toHaveTextContent("Alice");
  });

  it("removes a single filter when its chip remove button is clicked", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    await user.click(screen.getByTestId("filter-chip-remove-name"));
    expect(screen.queryByTestId("filter-chip-name")).toBeNull();
    expect(names()).toEqual(["Alice", "Bob"]);
  });

  it("clears every filter via Clear all", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    await user.click(screen.getByTestId("datatable-clear-filters"));
    expect(screen.queryByTestId("filter-chip-name")).toBeNull();
    expect(names()).toEqual(["Alice", "Bob"]);
  });

  it("shows the active filter count and fires onOpenFilters", async () => {
    const user = userEvent.setup();
    const onOpenFilters = vi.fn();
    render(<Harness onOpenFilters={onOpenFilters} />);
    expect(screen.getByTestId("datatable-filter-count")).toHaveTextContent("1");
    await user.click(screen.getByTestId("datatable-filters-trigger"));
    expect(onOpenFilters).toHaveBeenCalledTimes(1);
  });

  it("renders slotted action children", () => {
    render(
      <Harness>
        <button data-testid="toolbar-action">Action</button>
      </Harness>,
    );
    expect(screen.getByTestId("toolbar-action")).toBeInTheDocument();
  });

  it("fires onRefresh when the refresh button is clicked", async () => {
    const user = userEvent.setup();
    const onRefresh = vi.fn();
    render(<Harness onRefresh={onRefresh} />);
    await user.click(screen.getByTestId("datatable-refresh"));
    expect(onRefresh).toHaveBeenCalledTimes(1);
  });
});
