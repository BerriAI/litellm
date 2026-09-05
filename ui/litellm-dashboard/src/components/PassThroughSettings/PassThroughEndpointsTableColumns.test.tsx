import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { flexRender, getCoreRowModel, useReactTable, type ColumnDef } from "@tanstack/react-table";

import { getPassThroughEndpointsTableColumns } from "./PassThroughEndpointsTableColumns";
import type { passThroughItem } from "./PassThroughSettings";

const dbEndpoint: passThroughItem = {
  id: "db-endpoint-id",
  path: "/db-endpoint",
  target: "https://example.com/db",
  headers: {},
};

const configEndpoint: passThroughItem = {
  id: "config-endpoint-id",
  path: "/config-endpoint",
  target: "https://example.com/config",
  headers: {},
  is_from_config: true,
};

const defaultDeps = {
  onEndpointClick: vi.fn(),
  onDeleteClick: vi.fn(),
};

// Renders the column definitions through a real TanStack table so each `cell`
// renderer runs exactly as the DataTable runs it.
function TableHarness({ columns, data }: { columns: ColumnDef<passThroughItem>[]; data: passThroughItem[] }) {
  const table = useReactTable({ columns, data, getCoreRowModel: getCoreRowModel() });
  return (
    <table>
      <tbody>
        {table.getRowModel().rows.map((row) => (
          <tr key={row.id}>
            {row.getVisibleCells().map((cell) => (
              <td key={cell.id}>{flexRender(cell.column.columnDef.cell, cell.getContext())}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

const renderTable = (data: passThroughItem[]) =>
  render(<TableHarness columns={getPassThroughEndpointsTableColumns(defaultDeps)} data={data} />);

describe("getPassThroughEndpointsTableColumns", () => {
  it("shows the source of each endpoint", () => {
    renderTable([dbEndpoint, configEndpoint]);

    expect(screen.getByText("Database")).toBeInTheDocument();
    expect(screen.getByText("Config file")).toBeInTheDocument();
  });

  it("disables Edit and Delete for config-file-defined endpoints", async () => {
    const user = userEvent.setup();
    renderTable([configEndpoint]);

    await user.click(screen.getByTestId("endpoint-actions-config-endpoint-id"));

    const editItem = await screen.findByTestId("endpoint-action-edit");
    const deleteItem = screen.getByTestId("endpoint-action-delete");
    expect(editItem).toHaveAttribute("aria-disabled", "true");
    expect(deleteItem).toHaveAttribute("aria-disabled", "true");
  });

  it("keeps Edit and Delete enabled for DB endpoints", async () => {
    const user = userEvent.setup();
    renderTable([dbEndpoint]);

    await user.click(screen.getByTestId("endpoint-actions-db-endpoint-id"));

    const editItem = await screen.findByTestId("endpoint-action-edit");
    const deleteItem = screen.getByTestId("endpoint-action-delete");
    expect(editItem).not.toHaveAttribute("aria-disabled", "true");
    expect(deleteItem).not.toHaveAttribute("aria-disabled", "true");

    await user.click(deleteItem);
    expect(defaultDeps.onDeleteClick).toHaveBeenCalledWith("db-endpoint-id");
  });
});
