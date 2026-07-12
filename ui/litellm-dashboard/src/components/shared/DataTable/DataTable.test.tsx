import type { ColumnDef, ExpandedState } from "@tanstack/react-table";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";

import { DataTable } from "./DataTable";
import { DataTableSortHeader } from "./DataTableSortHeader";
import { DataTableViewOptions } from "./DataTableViewOptions";

interface Person {
  id: string;
  name: string;
  email: string;
  flagged?: boolean;
}

function person(id: string, name: string, flagged = false): Person {
  return { id, name, email: `${name.toLowerCase()}@x.io`, flagged };
}

const names = (): (string | null)[] => screen.getAllByTestId("name-cell").map((el) => el.textContent);

const nameCellColumns: ColumnDef<Person, unknown>[] = [
  {
    accessorKey: "name",
    header: "Name",
    cell: ({ row }) => <span data-testid="name-cell">{row.original.name}</span>,
  },
];

const filterableColumns: ColumnDef<Person, unknown>[] = [
  {
    accessorKey: "name",
    header: "Name",
    meta: { title: "Name" },
    filterFn: (row, columnId, value) => row.getValue<string>(columnId) === value,
    cell: ({ row }) => <span data-testid="name-cell">{row.original.name}</span>,
  },
];

const headerCycleColumns: ColumnDef<Person, unknown>[] = [
  {
    accessorKey: "name",
    header: ({ column }) => <DataTableSortHeader column={column} title="Name" variant="header-cycle" />,
    cell: ({ row }) => <span data-testid="name-cell">{row.original.name}</span>,
  },
];

const dropdownSortColumns: ColumnDef<Person, unknown>[] = [
  {
    accessorKey: "name",
    header: ({ column }) => <DataTableSortHeader column={column} title="Name" variant="dropdown-tristate" />,
    cell: ({ row }) => <span data-testid="name-cell">{row.original.name}</span>,
  },
];

const nameEmailColumns: ColumnDef<Person, unknown>[] = [
  {
    accessorKey: "name",
    header: "Name",
    cell: ({ row }) => <span data-testid="name-cell">{row.original.name}</span>,
  },
  {
    accessorKey: "email",
    header: "Email",
    cell: ({ row }) => <span>{row.original.email}</span>,
  },
];

const pinnedColumns: ColumnDef<Person, unknown>[] = [
  {
    accessorKey: "name",
    header: "Name",
    cell: ({ row }) => <span data-testid="name-cell">{row.original.name}</span>,
    meta: { pinned: "left" },
  },
  {
    accessorKey: "email",
    header: "Email",
    cell: ({ row }) => <span>{row.original.email}</span>,
  },
];

const rowClickColumns: ColumnDef<Person, unknown>[] = [
  {
    accessorKey: "name",
    header: "Name",
    cell: ({ row }) => <span data-testid="name-cell">{row.original.name}</span>,
  },
  {
    id: "actions",
    header: "Actions",
    cell: () => (
      <div>
        <button data-testid="row-button">Act</button>
        <input data-testid="row-input" aria-label="row input" />
      </div>
    ),
  },
];

const expansionColumns: ColumnDef<Person, unknown>[] = [
  {
    id: "expander",
    header: "",
    cell: ({ row }) => (
      <button data-testid={`expand-${row.id}`} onClick={() => row.toggleExpanded()}>
        toggle
      </button>
    ),
  },
  {
    accessorKey: "name",
    header: "Name",
    cell: ({ row }) => <span data-testid="name-cell">{row.original.name}</span>,
  },
];

const CHARLIE_ALICE_BOB: Person[] = [person("c", "Charlie"), person("a", "Alice"), person("b", "Bob")];

describe("DataTable sorting", () => {
  it("client mode reorders rows when the sort header is clicked", async () => {
    const user = userEvent.setup();
    render(<DataTable data={CHARLIE_ALICE_BOB} columns={headerCycleColumns} sortingMode="client" />);

    expect(names()).toEqual(["Charlie", "Alice", "Bob"]);
    await user.click(screen.getByTestId("sort-header-name"));
    expect(names()).toEqual(["Alice", "Bob", "Charlie"]);
  });

  it("server mode fires the callback but never reorders locally", async () => {
    const user = userEvent.setup();
    const onSortingChange = vi.fn();
    render(
      <DataTable
        data={CHARLIE_ALICE_BOB}
        columns={headerCycleColumns}
        sortingMode="server"
        sorting={[{ id: "name", desc: false }]}
        onSortingChange={onSortingChange}
      />,
    );

    // sorting state says ascending, but server mode must render data as given
    expect(names()).toEqual(["Charlie", "Alice", "Bob"]);
    await user.click(screen.getByTestId("sort-header-name"));
    expect(onSortingChange).toHaveBeenCalledTimes(1);
    expect(names()).toEqual(["Charlie", "Alice", "Bob"]);
  });

  it("dropdown-tristate variant sorts ascending, descending, then resets", async () => {
    const user = userEvent.setup();
    render(<DataTable data={CHARLIE_ALICE_BOB} columns={dropdownSortColumns} sortingMode="client" />);

    await user.click(screen.getByTestId("sort-trigger-name"));
    await user.click(await screen.findByText("Ascending"));
    expect(names()).toEqual(["Alice", "Bob", "Charlie"]);

    await user.click(screen.getByTestId("sort-trigger-name"));
    await user.click(await screen.findByText("Descending"));
    expect(names()).toEqual(["Charlie", "Bob", "Alice"]);

    await user.click(screen.getByTestId("sort-trigger-name"));
    await user.click(await screen.findByText("Reset"));
    expect(names()).toEqual(["Charlie", "Alice", "Bob"]);
  });
});

describe("DataTable pagination", () => {
  const fivePeople: Person[] = Array.from({ length: 5 }, (_, i) => person(String(i), `P${i}`));

  it("client mode slices rows and advances pages", async () => {
    const user = userEvent.setup();
    render(<DataTable data={fivePeople} columns={nameCellColumns} paginationMode="client" pageSizeOptions={[2]} />);

    expect(names()).toEqual(["P0", "P1"]);
    expect(screen.getByTestId("pagination-range")).toHaveTextContent("Showing 1-2 of 5");

    await user.click(screen.getByTestId("pagination-next"));
    expect(names()).toEqual(["P2", "P3"]);
    expect(screen.getByTestId("pagination-range")).toHaveTextContent("Showing 3-4 of 5");
  });

  it("server mode shows X-Y of Z from rowCount and does NOT slice the given rows", async () => {
    const user = userEvent.setup();
    const onPaginationChange = vi.fn();
    const pageSlice: Person[] = [person("10", "P10"), person("11", "P11"), person("12", "P12")];
    render(
      <DataTable
        data={pageSlice}
        columns={nameCellColumns}
        paginationMode="server"
        pagination={{ pageIndex: 1, pageSize: 10 }}
        rowCount={25}
        onPaginationChange={onPaginationChange}
      />,
    );

    expect(names()).toEqual(["P10", "P11", "P12"]);
    expect(screen.getByTestId("pagination-range")).toHaveTextContent("Showing 11-20 of 25");

    await user.click(screen.getByTestId("pagination-next"));
    expect(onPaginationChange).toHaveBeenCalledTimes(1);
  });
});

describe("DataTable filtering", () => {
  it("client mode filters rows by columnFilters", () => {
    const { rerender } = render(
      <DataTable
        data={CHARLIE_ALICE_BOB}
        columns={filterableColumns}
        filterMode="client"
        columnFilters={[]}
        onColumnFiltersChange={vi.fn()}
      />,
    );
    expect(names()).toEqual(["Charlie", "Alice", "Bob"]);

    rerender(
      <DataTable
        data={CHARLIE_ALICE_BOB}
        columns={filterableColumns}
        filterMode="client"
        columnFilters={[{ id: "name", value: "Alice" }]}
        onColumnFiltersChange={vi.fn()}
      />,
    );
    expect(names()).toEqual(["Alice"]);
  });

  it("client global filter matches substrings across columns", () => {
    const { rerender } = render(
      <DataTable
        data={CHARLIE_ALICE_BOB}
        columns={nameEmailColumns}
        filterMode="client"
        globalFilter=""
        onGlobalFilterChange={vi.fn()}
      />,
    );
    expect(names()).toEqual(["Charlie", "Alice", "Bob"]);

    rerender(
      <DataTable
        data={CHARLIE_ALICE_BOB}
        columns={nameEmailColumns}
        filterMode="client"
        globalFilter="ali"
        onGlobalFilterChange={vi.fn()}
      />,
    );
    expect(names()).toEqual(["Alice"]);
  });

  it("server mode never filters locally even when columnFilters is set", () => {
    render(
      <DataTable
        data={CHARLIE_ALICE_BOB}
        columns={filterableColumns}
        filterMode="server"
        columnFilters={[{ id: "name", value: "Alice" }]}
        onColumnFiltersChange={vi.fn()}
      />,
    );
    expect(names()).toEqual(["Charlie", "Alice", "Bob"]);
  });

  it("throws when server filtering is missing required props", () => {
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    expect(() => render(<DataTable data={[]} columns={filterableColumns} filterMode="server" />)).toThrow(
      /filterMode='server'/,
    );
    spy.mockRestore();
  });
});

describe("DataTable loading", () => {
  it("renders skeleton rows while loading and real rows once loaded", () => {
    const { rerender } = render(<DataTable data={CHARLIE_ALICE_BOB} columns={nameCellColumns} isLoading />);
    expect(screen.getAllByTestId("skeleton-row").length).toBeGreaterThan(0);
    expect(screen.queryByTestId("name-cell")).toBeNull();

    rerender(<DataTable data={CHARLIE_ALICE_BOB} columns={nameCellColumns} />);
    expect(screen.queryAllByTestId("skeleton-row")).toHaveLength(0);
    expect(names()).toEqual(["Charlie", "Alice", "Bob"]);
  });

  it("varies skeleton shape and width per column instead of one fixed bar", () => {
    const columns: ColumnDef<Person, unknown>[] = [
      { accessorKey: "name", header: "Name", meta: { skeleton: "twoLine" }, cell: () => null },
      { accessorKey: "email", header: "Email", cell: () => null },
    ];
    render(<DataTable data={CHARLIE_ALICE_BOB} columns={columns} isLoading />);

    const firstRow = screen.getAllByTestId("skeleton-row").at(0);
    expect(firstRow).toBeDefined();
    const bars = Array.from(firstRow?.querySelectorAll('[data-slot="skeleton"]') ?? []);

    // twoLine column contributes a main + sub bar (2); the text column contributes 1
    expect(bars).toHaveLength(3);
    // per-column widths differ instead of every cell sharing one fixed width
    expect(new Set(bars.map((bar) => bar.className)).size).toBeGreaterThan(1);
  });

  it("renders shape-specific skeletons for badge, chips, and meter columns", () => {
    const columns: ColumnDef<Person, unknown>[] = [
      { id: "badge", header: "Badge", meta: { skeleton: "badge" }, cell: () => null },
      { id: "chips", header: "Chips", meta: { skeleton: "chips" }, cell: () => null },
      { id: "meter", header: "Meter", meta: { skeleton: "meter" }, cell: () => null },
    ];
    render(<DataTable data={CHARLIE_ALICE_BOB} columns={columns} isLoading />);

    const firstRow = screen.getAllByTestId("skeleton-row").at(0);
    const cells = Array.from(firstRow?.querySelectorAll("td") ?? []);
    const barsIn = (cell: Element | undefined) => cell?.querySelectorAll('[data-slot="skeleton"]').length ?? 0;

    // badge = a single pill, chips = three pills, meter = value bar + track bar
    expect(barsIn(cells[0])).toBe(1);
    expect(cells[0]?.querySelector('[data-slot="skeleton"]')?.className).toContain("rounded-full");
    expect(barsIn(cells[1])).toBe(3);
    expect(barsIn(cells[2])).toBe(2);
  });
});

describe("DataTable column visibility", () => {
  it("hides a column when toggled off in the view-options menu", async () => {
    const user = userEvent.setup();
    const { container } = render(
      <DataTable
        data={CHARLIE_ALICE_BOB}
        columns={nameEmailColumns}
        toolbar={(table) => <DataTableViewOptions table={table} />}
      />,
    );

    expect(container.querySelector('th[data-header-id="email"]')).not.toBeNull();
    await user.click(screen.getByTestId("view-options-trigger"));
    await user.click(await screen.findByTestId("view-option-email"));
    await waitFor(() => expect(container.querySelector('th[data-header-id="email"]')).toBeNull());

    await user.click(screen.getByTestId("view-option-email"));
    await waitFor(() => expect(container.querySelector('th[data-header-id="email"]')).not.toBeNull());
  });

  it("omits columns that opt out of hiding from the menu", async () => {
    const user = userEvent.setup();
    const columns: ColumnDef<Person, unknown>[] = [
      {
        accessorKey: "name",
        header: "Name",
        enableHiding: false,
        cell: ({ row }) => <span data-testid="name-cell">{row.original.name}</span>,
      },
      {
        accessorKey: "email",
        header: "Email",
        cell: ({ row }) => <span>{row.original.email}</span>,
      },
    ];
    render(
      <DataTable
        data={CHARLIE_ALICE_BOB}
        columns={columns}
        toolbar={(table) => <DataTableViewOptions table={table} />}
      />,
    );

    await user.click(screen.getByTestId("view-options-trigger"));
    expect(await screen.findByTestId("view-option-email")).toBeInTheDocument();
    expect(screen.queryByTestId("view-option-name")).toBeNull();
  });
});

describe("DataTable pinned columns", () => {
  it("applies sticky positioning to a pinned column only", () => {
    const { container } = render(<DataTable data={CHARLIE_ALICE_BOB} columns={pinnedColumns} />);

    const pinnedHead = container.querySelector<HTMLElement>('th[data-header-id="name"]');
    const normalHead = container.querySelector<HTMLElement>('th[data-header-id="email"]');

    expect(pinnedHead?.style.position).toBe("sticky");
    expect(pinnedHead?.style.left).toBe("0px");
    expect(normalHead?.style.position).toBe("");
  });
});

describe("DataTable row click guard", () => {
  it("fires onRowClick from a plain cell but not from interactive elements", async () => {
    const user = userEvent.setup();
    const onRowClick = vi.fn();
    render(<DataTable data={[person("a", "Alice")]} columns={rowClickColumns} onRowClick={onRowClick} />);

    await user.click(screen.getByTestId("name-cell"));
    expect(onRowClick).toHaveBeenCalledTimes(1);
    expect(onRowClick).toHaveBeenCalledWith(expect.objectContaining({ id: "a" }));

    await user.click(screen.getByTestId("row-button"));
    expect(onRowClick).toHaveBeenCalledTimes(1);

    await user.click(screen.getByTestId("row-input"));
    expect(onRowClick).toHaveBeenCalledTimes(1);
  });
});

describe("DataTable expansion", () => {
  const subComponent = ({ row }: { row: { original: Person } }) => (
    <div data-testid="sub-row">details for {row.original.name}</div>
  );

  it("toggles the sub-row in uncontrolled mode", async () => {
    const user = userEvent.setup();
    render(
      <DataTable
        data={[person("a", "Alice")]}
        columns={expansionColumns}
        getRowId={(row) => row.id}
        getRowCanExpand={() => true}
        renderSubComponent={subComponent}
      />,
    );

    expect(screen.queryByTestId("sub-row")).not.toBeInTheDocument();
    await user.click(screen.getByTestId("expand-a"));
    expect(screen.getByTestId("sub-row")).toBeInTheDocument();
    await user.click(screen.getByTestId("expand-a"));
    expect(screen.queryByTestId("sub-row")).not.toBeInTheDocument();
  });

  it("toggles the sub-row in controlled mode driven by parent state", async () => {
    const user = userEvent.setup();
    const Harness = () => {
      const [expanded, setExpanded] = useState<ExpandedState>({});
      return (
        <DataTable
          data={[person("a", "Alice")]}
          columns={expansionColumns}
          getRowId={(row) => row.id}
          expanded={expanded}
          onExpandedChange={setExpanded}
          getRowCanExpand={() => true}
          renderSubComponent={subComponent}
        />
      );
    };
    render(<Harness />);

    expect(screen.queryByTestId("sub-row")).not.toBeInTheDocument();
    await user.click(screen.getByTestId("expand-a"));
    expect(screen.getByTestId("sub-row")).toBeInTheDocument();
    await user.click(screen.getByTestId("expand-a"));
    expect(screen.queryByTestId("sub-row")).not.toBeInTheDocument();
  });

  it("stays collapsed in controlled mode when the parent ignores the change", async () => {
    const user = userEvent.setup();
    const onExpandedChange = vi.fn();
    render(
      <DataTable
        data={[person("a", "Alice")]}
        columns={expansionColumns}
        getRowId={(row) => row.id}
        expanded={{}}
        onExpandedChange={onExpandedChange}
        getRowCanExpand={() => true}
        renderSubComponent={subComponent}
      />,
    );

    await user.click(screen.getByTestId("expand-a"));
    expect(onExpandedChange).toHaveBeenCalledTimes(1);
    expect(screen.queryByTestId("sub-row")).not.toBeInTheDocument();
  });
});

describe("DataTable row styling and footer", () => {
  it("applies rowClassName to the matching row only", () => {
    const data = [person("a", "Alice", true), person("b", "Bob", false)];
    const { container } = render(
      <DataTable
        data={data}
        columns={nameCellColumns}
        getRowId={(row) => row.id}
        rowClassName={(row) => (row.original.flagged ? "flagged-row" : "")}
      />,
    );

    expect(container.querySelector('tr[data-row-id="a"]')?.className).toContain("flagged-row");
    expect(container.querySelector('tr[data-row-id="b"]')?.className).not.toContain("flagged-row");
  });

  it("renders the footer slot inside a tfoot element", () => {
    render(
      <DataTable
        data={CHARLIE_ALICE_BOB}
        columns={nameCellColumns}
        footer={() => (
          <tr data-testid="footer-row">
            <td>Total: 3</td>
          </tr>
        )}
      />,
    );

    expect(screen.getByTestId("footer-row").closest("tfoot")).not.toBeNull();
  });
});

describe("DataTable layout", () => {
  it("exposes resize handles with stable selectors only when resizing is enabled", () => {
    const { container, rerender } = render(
      <DataTable data={CHARLIE_ALICE_BOB} columns={nameEmailColumns} enableColumnResizing />,
    );
    expect(container.querySelectorAll("[data-resizer][data-header-id]").length).toBe(2);

    rerender(<DataTable data={CHARLIE_ALICE_BOB} columns={nameEmailColumns} />);
    expect(container.querySelectorAll("[data-resizer]").length).toBe(0);
  });

  it("makes the header sticky and constrains body height when maxBodyHeight is set", () => {
    const { container } = render(<DataTable data={CHARLIE_ALICE_BOB} columns={nameEmailColumns} maxBodyHeight={240} />);
    expect(container.querySelector("thead")?.className).toContain("sticky");
    const scroller = container.querySelector('[data-slot="table-container"]')?.parentElement as HTMLElement;
    expect(scroller.style.maxHeight).toBe("240px");
  });
});

describe("DataTable misconfiguration guards", () => {
  it("throws when server sorting is missing required props", () => {
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    expect(() => render(<DataTable data={[]} columns={nameCellColumns} sortingMode="server" />)).toThrow(
      /sortingMode='server'/,
    );
    spy.mockRestore();
  });

  it("throws when server pagination is missing required props", () => {
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    expect(() => render(<DataTable data={[]} columns={nameCellColumns} paginationMode="server" />)).toThrow(
      /paginationMode='server'/,
    );
    spy.mockRestore();
  });

  it("throws when both defaultSorting and sorting are provided", () => {
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    expect(() =>
      render(
        <DataTable
          data={[]}
          columns={nameCellColumns}
          defaultSorting={[{ id: "name", desc: false }]}
          sorting={[{ id: "name", desc: false }]}
        />,
      ),
    ).toThrow(/defaultSorting/);
    spy.mockRestore();
  });
});
