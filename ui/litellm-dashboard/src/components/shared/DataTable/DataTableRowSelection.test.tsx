import type { ColumnDef, Row, RowSelectionState } from "@tanstack/react-table";
import { act, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

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

interface HarnessProps {
  onRowClick?: (row: Model) => void;
  enableRowSelection?: boolean | ((row: Row<Model>) => boolean);
}

function ControlledHarness(props: HarnessProps) {
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
        {...props}
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

    it("keeps scrolling and selecting while the pointer rests near the bottom edge, and stops on release", () => {
      vi.useFakeTimers({ toFake: ["requestAnimationFrame", "cancelAnimationFrame"] });
      const scrolled = { value: false };
      const scrollBy = vi.spyOn(window, "scrollBy").mockImplementation(() => {
        scrolled.value = true;
      });
      const middle = window.innerHeight / 2;
      const nearBottom = window.innerHeight - 5;
      const rowUnderPointer = (y: number) => {
        if (y === middle) return "m1";
        return scrolled.value ? "m3" : "m2";
      };
      Object.defineProperty(document, "elementFromPoint", {
        configurable: true,
        value: (_x: number, y: number) => rowBox(rowUnderPointer(y)),
      });
      render(<ControlledHarness />);

      pressRow("m1", middle);
      act(() => {
        vi.advanceTimersToNextFrame();
      });
      expect(scrollBy).not.toHaveBeenCalled();

      moveTo(nearBottom);
      expectKeys("m1,m2");

      act(() => {
        vi.advanceTimersToNextFrame();
        vi.advanceTimersToNextFrame();
      });
      expect(scrollBy).toHaveBeenCalledTimes(2);
      expect(scrollBy.mock.calls[0][1]).toBeGreaterThan(0);
      expectKeys("m1,m2,m3");

      fireEvent.pointerCancel(document);
      scrollBy.mockClear();
      act(() => {
        vi.advanceTimersToNextFrame();
      });
      expect(scrollBy).not.toHaveBeenCalled();

      scrollBy.mockRestore();
      vi.useRealTimers();
    });

    describe("from anywhere on a clickable row", () => {
      const pressCell = (name: string, y: number, button = 0) =>
        fireEvent.pointerDown(screen.getByText(name), { button, clientX: 200, clientY: y });

      it("starts selecting only once the pointer reaches another row, and does not open the row on release", () => {
        const onRowClick = vi.fn();
        render(<ControlledHarness onRowClick={onRowClick} enableRowSelection />);

        pressCell("Alpha", 10);
        moveTo(10);
        expectKeys("");
        expect(document.body).not.toHaveStyle({ userSelect: "none" });

        moveTo(30);
        expectKeys("m1,m2,m3");
        expect(document.body).toHaveStyle({ userSelect: "none" });

        moveTo(10);
        expectKeys("m1");
        fireEvent.pointerUp(document);
        fireEvent.click(screen.getByText("Alpha"));

        expect(onRowClick).not.toHaveBeenCalled();
        expect(document.body).not.toHaveStyle({ userSelect: "none" });

        pressCell("Alpha", 10);
        fireEvent.pointerUp(document);
        fireEvent.click(screen.getByText("Alpha"));
        expect(onRowClick).toHaveBeenCalledWith(data[0]);
      });

      it("leaves drags that start on a control inside the row to that control", () => {
        const onRowClick = vi.fn();
        render(
          <DataTable
            data={data}
            columns={[
              ...columns,
              {
                id: "open",
                header: "Open",
                cell: ({ row }) => <button type="button">Open {row.original.name}</button>,
              },
            ]}
            getRowId={(row) => row.id}
            enableRowSelection
            onRowClick={onRowClick}
            toolbar={(table) => <span data-testid="count">{table.getSelectedRowModel().rows.length}</span>}
          />,
        );

        fireEvent.pointerDown(screen.getByRole("button", { name: "Open Alpha" }), {
          button: 0,
          clientX: 200,
          clientY: 10,
        });
        moveTo(30);
        fireEvent.pointerUp(document);

        expect(selectedCount()).toHaveTextContent("0");
      });

      it("still opens the row on a plain click", () => {
        const onRowClick = vi.fn();
        render(<ControlledHarness onRowClick={onRowClick} enableRowSelection />);

        pressCell("Beta", 20);
        fireEvent.pointerUp(document);
        fireEvent.click(screen.getByText("Beta"));

        expect(onRowClick).toHaveBeenCalledWith(data[1]);
        expectKeys("");
      });

      it("clears highlighted text once the drag turns into a selection", () => {
        render(<ControlledHarness onRowClick={vi.fn()} enableRowSelection />);

        pressCell("Alpha", 10);
        window.getSelection()?.selectAllChildren(screen.getByText("Alpha"));
        moveTo(20);
        fireEvent.pointerUp(document);

        expect(window.getSelection()?.isCollapsed).toBe(true);
      });

      it("ignores secondary buttons and rows that cannot be selected", () => {
        render(<ControlledHarness onRowClick={vi.fn()} enableRowSelection={(row) => row.original.id !== "m1"} />);

        pressCell("Beta", 20, 2);
        moveTo(30);
        fireEvent.pointerUp(document);
        expectKeys("");

        pressCell("Alpha", 10);
        moveTo(30);
        fireEvent.pointerUp(document);
        expectKeys("");
      });

      it("does nothing on tables without row selection or without click-to-open rows", () => {
        const { unmount } = render(<ControlledHarness onRowClick={vi.fn()} />);
        pressCell("Alpha", 10);
        moveTo(30);
        fireEvent.pointerUp(document);
        expectKeys("");
        unmount();

        render(<ControlledHarness enableRowSelection />);
        pressCell("Alpha", 10);
        moveTo(30);
        fireEvent.pointerUp(document);
        expectKeys("");
      });
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
