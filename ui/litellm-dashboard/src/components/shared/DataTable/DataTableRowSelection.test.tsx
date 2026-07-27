import type { ColumnDef, RowSelectionState } from "@tanstack/react-table";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it } from "vitest";

import { createSelectionColumn, DataTable, validateDataTableConfig } from "./index";

interface Model {
  id: string;
  name: string;
}

const data: Model[] = [
  { id: "m1", name: "Alpha" },
  { id: "m2", name: "Beta" },
  { id: "m3", name: "Gamma" },
];

const columns: ColumnDef<Model, unknown>[] = [
  createSelectionColumn<Model>({ rowAriaLabel: (row) => `Select ${row.original.name}` }),
  { id: "name", accessorKey: "name", header: "Name", enableSorting: false },
];

const selectAll = () => screen.getByTestId("datatable-select-all");
const rowBox = (id: string) => screen.getByTestId(`datatable-select-row-${id}`);
const selectedCount = () => screen.getByTestId("count");

function ControlledHarness() {
  const [rowSelection, setRowSelection] = useState<RowSelectionState>({});

  return (
    <>
      <span data-testid="keys">
        {Object.keys(rowSelection)
          .filter((key) => rowSelection[key])
          .sort()
          .join(",")}
      </span>
      <button type="button" data-testid="clear" onClick={() => setRowSelection({})}>
        clear
      </button>
      <DataTable
        data={data}
        columns={columns}
        getRowId={(row) => row.id}
        rowSelection={rowSelection}
        onRowSelectionChange={setRowSelection}
      />
    </>
  );
}

describe("DataTable row selection", () => {
  it("supports uncontrolled per-row toggle, select-all, and indeterminate", async () => {
    const user = userEvent.setup();

    render(
      <DataTable
        data={data}
        columns={columns}
        getRowId={(row) => row.id}
        toolbar={(table) => <span data-testid="count">{table.getSelectedRowModel().rows.length}</span>}
      />,
    );

    expect(selectedCount()).toHaveTextContent("0");

    await user.click(rowBox("m1"));
    expect(selectedCount()).toHaveTextContent("1");
    expect(selectAll()).toHaveAttribute("aria-checked", "mixed");

    await user.click(selectAll());
    expect(selectedCount()).toHaveTextContent("3");
    expect(selectAll()).toHaveAttribute("aria-checked", "true");

    await user.click(selectAll());
    expect(selectedCount()).toHaveTextContent("0");
  });

  it("keys controlled selection by getRowId so the parent can map back to entities", async () => {
    const user = userEvent.setup();
    render(<ControlledHarness />);

    await user.click(rowBox("m2"));
    expect(screen.getByTestId("keys")).toHaveTextContent("m2");

    await user.click(rowBox("m3"));
    expect(screen.getByTestId("keys")).toHaveTextContent("m2,m3");
  });

  it("lets the parent clear the selection, the pattern an external pager needs", async () => {
    const user = userEvent.setup();
    render(<ControlledHarness />);

    await user.click(selectAll());
    expect(screen.getByTestId("keys")).toHaveTextContent("m1,m2,m3");

    await user.click(screen.getByTestId("clear"));
    expect(screen.getByTestId("keys")).toBeEmptyDOMElement();
    expect(rowBox("m1")).toHaveAttribute("aria-checked", "false");
  });

  it("respects an enableRowSelection predicate", async () => {
    const user = userEvent.setup();

    render(
      <DataTable
        data={data}
        columns={columns}
        getRowId={(row) => row.id}
        enableRowSelection={(row) => row.original.id !== "m2"}
        toolbar={(table) => <span data-testid="count">{table.getSelectedRowModel().rows.length}</span>}
      />,
    );

    expect(rowBox("m2")).toHaveAttribute("aria-disabled", "true");

    await user.click(rowBox("m2"));
    expect(selectedCount()).toHaveTextContent("0");

    await user.click(rowBox("m1"));
    expect(selectedCount()).toHaveTextContent("1");
  });

  it("rejects controlled rowSelection without onRowSelectionChange", () => {
    const errors = validateDataTableConfig<Model, unknown>({ data, columns, rowSelection: { m1: true } });

    expect(errors).toContain(
      "Controlled `rowSelection` requires `onRowSelectionChange`; without it selection changes are dropped.",
    );
  });

  it("does not complain when selection is left uncontrolled", () => {
    expect(validateDataTableConfig<Model, unknown>({ data, columns })).toHaveLength(0);
  });
});
