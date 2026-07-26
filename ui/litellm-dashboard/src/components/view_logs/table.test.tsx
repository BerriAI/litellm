import type { ColumnDef } from "@tanstack/react-table";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { DataTable } from "./table";

type Row = { request_id: string; a: string; b: string };

const data: Row[] = [{ request_id: "r1", a: "alpha", b: "beta" }];

const sizedColumns: ColumnDef<Row>[] = [
  { header: "A", accessorKey: "a", size: 120 },
  { header: "B", accessorKey: "b", size: 80 },
];

const unsizedColumns: ColumnDef<Row>[] = [
  { header: "A", accessorKey: "a" },
  { header: "B", accessorKey: "b" },
];

const expanderColumn: ColumnDef<Row> = {
  id: "expander",
  header: () => null,
  cell: ({ row }) =>
    row.getCanExpand() ? (
      <button
        onClick={row.getToggleExpandedHandler()}
        aria-label={`${row.getIsExpanded() ? "collapse" : "expand"} ${row.original.request_id}`}
      >
        {row.getIsExpanded() ? "collapse" : "expand"}
      </button>
    ) : null,
};

describe("DataTable column sizing", () => {
  it("min-widths the table to the column total and sizes every cell when columns declare sizes", () => {
    render(<DataTable data={data} columns={sizedColumns} />);

    const table = screen.getByRole("table");
    expect(table.style.minWidth).toBe("200px");
    expect(table.style.width).toBe("");

    const headers = screen.getAllByRole("columnheader");
    expect(headers.map((h) => h.style.width)).toEqual(["120px", "80px"]);

    const cells = screen.getAllByRole("cell");
    expect(cells.map((c) => c.style.width)).toEqual(["120px", "80px"]);
  });

  it("leaves cells unsized and keeps the fluid table when no column declares a size", () => {
    render(<DataTable data={data} columns={unsizedColumns} />);

    const table = screen.getByRole("table");
    expect(table.style.width).toBe("");
    expect(table.style.minWidth).toBe("400px");

    for (const cell of [...screen.getAllByRole("columnheader"), ...screen.getAllByRole("cell")]) {
      expect(cell.style.width).toBe("");
    }
  });
});

describe("DataTable states", () => {
  it("shows the loading message instead of rows while loading", () => {
    render(<DataTable data={data} columns={unsizedColumns} isLoading loadingMessage="Fetching things" />);

    expect(screen.getByText("Fetching things")).toBeInTheDocument();
    expect(screen.queryByText("alpha")).not.toBeInTheDocument();
  });

  it("shows the no-data message when there are no rows", () => {
    render(<DataTable data={[]} columns={unsizedColumns} noDataMessage="Nothing here" />);

    expect(screen.getByText("Nothing here")).toBeInTheDocument();
  });

  it("falls back to generic loading and empty defaults", () => {
    const { rerender } = render(<DataTable data={data} columns={unsizedColumns} isLoading />);
    expect(screen.getByText("Loading...")).toBeInTheDocument();

    rerender(<DataTable data={[]} columns={unsizedColumns} />);
    expect(screen.getByText("No results")).toBeInTheDocument();
  });

  it("suppresses the primitive's row hover on loading, empty, and expansion placeholder rows", async () => {
    const user = userEvent.setup();
    const { rerender } = render(<DataTable data={data} columns={unsizedColumns} isLoading />);
    expect(screen.getByText("Loading...").closest("tr")).toHaveClass("hover:bg-transparent");

    rerender(<DataTable data={[]} columns={unsizedColumns} />);
    expect(screen.getByText("No results").closest("tr")).toHaveClass("hover:bg-transparent");

    rerender(
      <DataTable
        data={data}
        columns={[expanderColumn, ...unsizedColumns]}
        getRowCanExpand={() => true}
        renderSubComponent={({ row }) => <div>details for {row.original.request_id}</div>}
      />,
    );
    await user.click(screen.getByRole("button", { name: "expand r1" }));
    expect(screen.getByText("details for r1").closest("tr")).toHaveClass("hover:bg-transparent");
    expect(screen.getByText("alpha").closest("tr")).not.toHaveClass("hover:bg-transparent");
  });

  it("renders row data through plain TanStack column defs, including custom cell renderers", () => {
    const columns: ColumnDef<Row>[] = [
      { header: "A", accessorKey: "a" },
      { header: "B", cell: ({ row }) => <span>custom:{row.original.b}</span> },
    ];
    render(<DataTable data={data} columns={columns} />);

    expect(screen.getByText("alpha")).toBeInTheDocument();
    expect(screen.getByText("custom:beta")).toBeInTheDocument();
  });

  it("clips the table to the rounded wrapper so the header band cannot bleed past the corners", () => {
    const { container } = render(<DataTable data={data} columns={unsizedColumns} />);

    const wrapper = container.firstElementChild;
    expect(wrapper).toHaveClass("rounded-lg", "overflow-hidden");
  });

  it("right-aligns headers and cells with tabular figures for numeric meta columns", () => {
    const columns: ColumnDef<Row>[] = [
      { header: "A", accessorKey: "a" },
      { header: "B", accessorKey: "b", meta: { numeric: true } },
    ];
    render(<DataTable data={data} columns={columns} />);

    const headers = screen.getAllByRole("columnheader");
    expect(headers[1].querySelector("div")).toHaveClass("justify-end");
    expect(headers[0].querySelector("div")).not.toHaveClass("justify-end");

    const cells = screen.getAllByRole("cell");
    expect(cells[1]).toHaveClass("text-right", "tabular-nums");
    expect(cells[0]).not.toHaveClass("text-right");
  });
});

describe("DataTable row interaction", () => {
  it("fires onRowClick with the row's original data", async () => {
    const user = userEvent.setup();
    const onRowClick = vi.fn();
    render(<DataTable data={data} columns={unsizedColumns} onRowClick={onRowClick} />);

    await user.click(screen.getByText("alpha"));

    expect(onRowClick).toHaveBeenCalledExactlyOnceWith(data[0]);
  });
});

describe("DataTable expansion", () => {
  const rows: Row[] = [
    { request_id: "r1", a: "alpha", b: "beta" },
    { request_id: "r2", a: "gamma", b: "delta" },
  ];

  it("toggles the sub-component in a full-width cell (colspan path)", async () => {
    const user = userEvent.setup();
    render(
      <DataTable
        data={rows}
        columns={[expanderColumn, ...unsizedColumns]}
        getRowCanExpand={() => true}
        renderSubComponent={({ row }) => <div>details for {row.original.request_id}</div>}
      />,
    );

    expect(screen.queryByText("details for r1")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "expand r1" }));
    const details = screen.getByText("details for r1");
    expect(details).toBeInTheDocument();
    expect(screen.queryByText("details for r2")).not.toBeInTheDocument();

    const detailCell = details.closest("td");
    expect(detailCell).toHaveAttribute("colspan", "3");

    await user.click(screen.getByRole("button", { name: "collapse r1" }));
    expect(screen.queryByText("details for r1")).not.toBeInTheDocument();
  });

  it("keeps expansion attached to the same row through data reorders when getRowId is injected", async () => {
    const user = userEvent.setup();
    const { rerender } = render(
      <DataTable
        data={rows}
        columns={[expanderColumn, ...unsizedColumns]}
        getRowId={(row) => row.request_id}
        getRowCanExpand={() => true}
        renderSubComponent={({ row }) => <div>details for {row.original.request_id}</div>}
      />,
    );

    await user.click(screen.getByRole("button", { name: "expand r1" }));
    expect(screen.getByText("details for r1")).toBeInTheDocument();

    rerender(
      <DataTable
        data={[...rows].reverse()}
        columns={[expanderColumn, ...unsizedColumns]}
        getRowId={(row) => row.request_id}
        getRowCanExpand={() => true}
        renderSubComponent={({ row }) => <div>details for {row.original.request_id}</div>}
      />,
    );

    expect(screen.getByText("details for r1")).toBeInTheDocument();
    expect(screen.queryByText("details for r2")).not.toBeInTheDocument();
  });

  it("does not expand rows when getRowCanExpand is missing even if a renderer is provided", () => {
    render(
      <DataTable
        data={rows}
        columns={[expanderColumn, ...unsizedColumns]}
        renderSubComponent={({ row }) => <div>details for {row.original.request_id}</div>}
      />,
    );

    expect(screen.queryByRole("button", { name: "expand r1" })).not.toBeInTheDocument();
  });
});

describe("DataTable sorting", () => {
  const rows: Row[] = [
    { request_id: "r1", a: "bravo", b: "2" },
    { request_id: "r2", a: "alpha", b: "1" },
    { request_id: "r3", a: "charlie", b: "3" },
  ];

  const firstColumnValues = () =>
    screen
      .getAllByRole("row")
      .slice(1)
      .map((row) => within(row).getAllByRole("cell")[0].textContent);

  it("leaves row order untouched when sorting is disabled", async () => {
    const user = userEvent.setup();
    render(<DataTable data={rows} columns={unsizedColumns} />);

    await user.click(screen.getByText("A"));

    expect(firstColumnValues()).toEqual(["bravo", "alpha", "charlie"]);
  });

  it("sorts ascending then descending on header clicks when enabled", async () => {
    const user = userEvent.setup();
    render(<DataTable data={rows} columns={unsizedColumns} enableSorting />);

    await user.click(screen.getByText("A"));
    expect(firstColumnValues()).toEqual(["alpha", "bravo", "charlie"]);

    await user.click(screen.getByText("A"));
    expect(firstColumnValues()).toEqual(["charlie", "bravo", "alpha"]);
  });
});
