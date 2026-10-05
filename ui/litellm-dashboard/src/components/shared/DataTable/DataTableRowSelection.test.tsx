import type { ColumnDef, RowSelectionState } from "@tanstack/react-table";
import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { createSelectionColumn, DataTable } from "./index";

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

  describe("drag to select", () => {
    const OTHER_TABLE_Y = 99;
    const rowAtY: Readonly<Record<number, string>> = { 10: "m1", 20: "m2", 30: "m3" };
    const expectKeys = (expected: string) =>
      expect(screen.getByTestId("keys")).toHaveTextContent(new RegExp(`^${expected}$`));
    const pressRow = (id: string, y: number, button = 0) =>
      fireEvent.pointerDown(rowBox(id), { button, clientX: 5, clientY: y });
    const moveTo = (y: number) => fireEvent.pointerMove(document, { clientX: 5, clientY: y });

    beforeEach(() => {
      Object.defineProperty(document, "elementFromPoint", {
        configurable: true,
        value: (_x: number, y: number) =>
          y === OTHER_TABLE_Y ? screen.getByTestId("other-table-row") : rowBox(rowAtY[y]),
      });
    });

    afterEach(() => {
      Reflect.deleteProperty(document, "elementFromPoint");
    });

    it("selects every row between the pressed row and the pointer, and shrinks when dragged back", () => {
      render(<ControlledHarness />);

      pressRow("m1", 10);
      expectKeys("m1");

      moveTo(30);
      expectKeys("m1,m2,m3");

      moveTo(20);
      expectKeys("m1,m2");

      fireEvent.pointerUp(document);
      moveTo(30);
      expectKeys("m1,m2");
    });

    it("deselects the dragged range when the pressed row was already selected", async () => {
      const user = userEvent.setup();
      render(<ControlledHarness />);
      await user.click(selectAll());

      pressRow("m3", 30);
      moveTo(20);
      fireEvent.pointerUp(document);

      expectKeys("m1");
    });

    it("ignores presses with a button other than the primary one", () => {
      render(<ControlledHarness />);

      pressRow("m1", 10, 2);
      moveTo(30);

      expectKeys("");
    });

    it("ignores rows in a different table that share an id", () => {
      render(
        <>
          <ControlledHarness />
          <table>
            <tbody>
              <tr data-row-id="m3">
                <td data-testid="other-table-row" />
              </tr>
            </tbody>
          </table>
        </>,
      );

      pressRow("m1", 10);
      moveTo(OTHER_TABLE_Y);

      expectKeys("m1");
    });

    it("skips rows that cannot be selected", () => {
      render(
        <DataTable
          data={data}
          columns={columns}
          getRowId={(row) => row.id}
          enableRowSelection={(row) => row.original.id !== "m2"}
          toolbar={(table) => <span data-testid="count">{table.getSelectedRowModel().rows.length}</span>}
        />,
      );

      pressRow("m1", 10);
      moveTo(30);
      fireEvent.pointerUp(document);

      expect(selectedCount()).toHaveTextContent("2");
      expect(rowBox("m2")).toHaveAttribute("aria-checked", "false");
    });
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
});
