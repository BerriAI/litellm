import {
  getCoreRowModel,
  getExpandedRowModel,
  useReactTable,
  type ColumnDef,
  type TableOptions,
} from "@tanstack/react-table";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it } from "vitest";

import { Inspector } from "./Inspector";
import { InspectorTable } from "./InspectorTable";

interface Node {
  readonly id: string;
  readonly children?: readonly Node[];
}

const TREE: Node[] = [
  { id: "a", children: [{ id: "a1" }, { id: "a2" }] },
  { id: "b", children: [{ id: "b1" }] },
  { id: "c" },
];

const COLUMNS: ColumnDef<Node>[] = [
  {
    id: "name",
    header: "Name",
    cell: ({ row }) => (
      <span>
        <InspectorTable.Indent row={row} toggleLabel={(expanded) => `${expanded ? "Hide" : "Show"} ${row.id}`} />
        {row.id}
      </span>
    ),
  },
];

function Tree() {
  const [selected, setSelected] = useState<Node | null>(null);
  const tableOptions: TableOptions<Node> = {
    data: TREE,
    columns: COLUMNS,
    getRowId: (node) => node.id,
    getSubRows: (node) => node.children?.slice(),
    initialState: { expanded: true },
    getCoreRowModel: getCoreRowModel(),
    getExpandedRowModel: getExpandedRowModel(),
  };
  const table = useReactTable(tableOptions);
  return (
    <Inspector.Root
      items={table.getRowModel().rows.map((row) => row.original)}
      itemKey={(node: Node) => node.id}
      selected={selected}
      onSelectedChange={setSelected}
      noun="node"
      storageKey="inspector-table-test"
    >
      <InspectorTable.Root table={table}>
        <InspectorTable.Grid aria-label="Nodes">
          <InspectorTable.Body<Node> rowHeight={() => 36}>
            {(row) => <InspectorTable.Row row={row} item={row.original} aria-label={row.id} />}
          </InspectorTable.Body>
        </InspectorTable.Grid>
      </InspectorTable.Root>
      <Inspector.Panel label="Node">{(node: Node) => <p>Open: {node.id}</p>}</Inspector.Panel>
    </Inspector.Root>
  );
}

const rowNames = () =>
  within(screen.getByRole("table", { name: "Nodes" }))
    .getAllByRole("row")
    .map((row) => row.getAttribute("aria-label"))
    .filter(Boolean);

describe("InspectorTable tree", () => {
  it("collapses and re-expands one parent's rows without touching its siblings", async () => {
    const user = userEvent.setup();
    render(<Tree />);
    expect(rowNames()).toEqual(["a", "a1", "a2", "b", "b1", "c"]);

    await user.click(screen.getByRole("button", { name: "Hide a" }));
    expect(rowNames()).toEqual(["a", "b", "b1", "c"]);

    await user.click(screen.getByRole("button", { name: "Show a" }));
    expect(rowNames()).toEqual(["a", "a1", "a2", "b", "b1", "c"]);
    expect(screen.queryByRole("button", { name: /c$/ })).not.toBeInTheDocument();
  });

  it("opens rows without toggling them, and J walks only the visible rows", async () => {
    const user = userEvent.setup();
    render(<Tree />);
    await user.click(screen.getByRole("button", { name: "Hide a" }));
    expect(screen.queryByText(/^Open:/)).not.toBeInTheDocument();

    await user.click(screen.getByRole("row", { name: "a" }));
    expect(screen.getByText("Open: a")).toBeInTheDocument();
    await user.keyboard("j");
    expect(screen.getByText("Open: b")).toBeInTheDocument();
    await user.keyboard("j");
    expect(screen.getByText("Open: b1")).toBeInTheDocument();
  });
});
