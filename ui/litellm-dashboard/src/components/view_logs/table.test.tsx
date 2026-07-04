import type { ColumnDef } from "@tanstack/react-table";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
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
