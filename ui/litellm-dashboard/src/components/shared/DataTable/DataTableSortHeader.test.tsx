import {
  type ColumnDef,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  type OnChangeFn,
  type SortingState,
  useReactTable,
} from "@tanstack/react-table";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";

import { DataTableSortHeader, type DataTableSortVariant } from "./DataTableSortHeader";

interface Item {
  name: string;
}

interface HarnessProps {
  variant: DataTableSortVariant;
  canSort?: boolean;
  onSortingChange?: OnChangeFn<SortingState>;
}

function SortHeaderHarness({ variant, canSort = true, onSortingChange }: HarnessProps) {
  const [sorting, setSorting] = useState<SortingState>([]);
  const columns: ColumnDef<Item, unknown>[] = [
    {
      accessorKey: "name",
      enableSorting: canSort,
      header: ({ column }) => <DataTableSortHeader column={column} title="Name" variant={variant} />,
    },
  ];
  const options = {
    data: [{ name: "x" }],
    columns,
    state: { sorting },
    onSortingChange: (updater: SortingState | ((prev: SortingState) => SortingState)) => {
      setSorting(updater);
      onSortingChange?.(updater);
    },
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  };
  const table = useReactTable(options);

  return (
    <table>
      <thead>
        {table.getHeaderGroups().map((headerGroup) => (
          <tr key={headerGroup.id}>
            {headerGroup.headers.map((header) => (
              <th key={header.id}>{flexRender(header.column.columnDef.header, header.getContext())}</th>
            ))}
          </tr>
        ))}
      </thead>
    </table>
  );
}

describe("DataTableSortHeader", () => {
  it("renders a plain label and no button when the column cannot sort", () => {
    render(<SortHeaderHarness variant="header-cycle" canSort={false} />);
    expect(screen.queryByTestId("sort-header-name")).toBeNull();
    expect(screen.getByText("Name")).toBeInTheDocument();
  });

  it("header-cycle indicator advances none -> asc -> desc on click", async () => {
    const user = userEvent.setup();
    render(<SortHeaderHarness variant="header-cycle" />);
    const indicator = () => screen.getByTestId("sort-header-name").querySelector("[data-sort-indicator]");

    expect(indicator()).toHaveAttribute("data-sort-indicator", "none");
    await user.click(screen.getByTestId("sort-header-name"));
    expect(indicator()).toHaveAttribute("data-sort-indicator", "asc");
    await user.click(screen.getByTestId("sort-header-name"));
    expect(indicator()).toHaveAttribute("data-sort-indicator", "desc");
  });

  it("dropdown-tristate sets ascending, descending, and reset from the menu", async () => {
    const user = userEvent.setup();
    render(<SortHeaderHarness variant="dropdown-tristate" />);

    await user.click(screen.getByTestId("sort-trigger-name"));
    await user.click(await screen.findByText("Descending"));
    expect(screen.getByTestId("sort-trigger-name").querySelector('[data-sort-indicator="desc"]')).not.toBeNull();

    await user.click(screen.getByTestId("sort-trigger-name"));
    await user.click(await screen.findByText("Ascending"));
    expect(screen.getByTestId("sort-trigger-name").querySelector('[data-sort-indicator="asc"]')).not.toBeNull();

    await user.click(screen.getByTestId("sort-trigger-name"));
    await user.click(await screen.findByText("Reset"));
    expect(screen.getByTestId("sort-trigger-name").querySelector('[data-sort-indicator="none"]')).not.toBeNull();
  });

  it("dropdown-tristate trigger stops the click from reaching an outer handler", async () => {
    const user = userEvent.setup();
    const onOuterClick = vi.fn();
    render(
      <div onClick={onOuterClick}>
        <SortHeaderHarness variant="dropdown-tristate" />
      </div>,
    );

    await user.click(screen.getByTestId("sort-trigger-name"));
    expect(onOuterClick).not.toHaveBeenCalled();
  });
});
