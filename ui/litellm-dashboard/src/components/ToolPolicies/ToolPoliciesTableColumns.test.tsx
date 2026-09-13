import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi } from "vitest";
import { flexRender, getCoreRowModel, useReactTable, type ColumnDef } from "@tanstack/react-table";
import { getToolPoliciesTableColumns } from "./ToolPoliciesTableColumns";
import type { ToolRow } from "@/components/networking";

const row: ToolRow = {
  tool_name: "search_docs",
  input_policy: "untrusted",
  output_policy: "trusted",
  call_count: 1234,
  team_id: "team-alpha",
  key_hash: "abc123def456",
  key_alias: "prod-key",
  user_agent: "litellm-python/1.0",
  created_at: "2026-03-04T10:00:00Z",
} as ToolRow;

const defaultDeps = {
  onSelectTool: vi.fn(),
  savingInput: new Set<string>(),
  savingOutput: new Set<string>(),
  onInputPolicyChange: vi.fn(),
  onOutputPolicyChange: vi.fn(),
};

// Renders the column definitions through a real TanStack table so each `cell`
// renderer runs exactly as the DataTable runs it.
function TableHarness({ columns, data }: { columns: ColumnDef<ToolRow>[]; data: ToolRow[] }) {
  const table = useReactTable({ columns, data, getCoreRowModel: getCoreRowModel() });
  return (
    <table>
      <tbody>
        {table.getRowModel().rows.map((r) => (
          <tr key={r.id}>
            {r.getVisibleCells().map((cell) => (
              <td key={cell.id}>{flexRender(cell.column.columnDef.cell, cell.getContext())}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

const renderTable = (deps = {}, data: ToolRow[] = [row]) =>
  render(<TableHarness columns={getToolPoliciesTableColumns({ ...defaultDeps, ...deps })} data={data} />);

describe("getToolPoliciesTableColumns", () => {
  it("defines the expected columns in order", () => {
    const columns = getToolPoliciesTableColumns(defaultDeps);

    expect(columns.map((c) => c.id)).toEqual([
      "created_at",
      "tool_name",
      "input_policy",
      "output_policy",
      "call_count",
      "team_id",
      "key_hash",
      "key_alias",
      "user_agent",
    ]);
  });

  it("renders the row's identifying fields", () => {
    renderTable();

    expect(screen.getByText("search_docs")).toBeInTheDocument();
    expect(screen.getByText("team-alpha")).toBeInTheDocument();
    expect(screen.getByText("prod-key")).toBeInTheDocument();
    expect(screen.getByText("litellm-python/1.0")).toBeInTheDocument();
  });

  it("formats the call count with thousands separators", () => {
    renderTable();

    expect(screen.getByText("1,234")).toBeInTheDocument();
  });

  it("renders a zero call count rather than a blank cell", () => {
    renderTable({}, [{ ...row, call_count: undefined } as ToolRow]);

    expect(screen.getByText("0")).toBeInTheDocument();
  });

  it("falls back to a dash for a missing key alias and user agent", () => {
    renderTable({}, [{ ...row, key_alias: undefined, user_agent: undefined } as ToolRow]);

    expect(screen.getAllByText("-").length).toBeGreaterThanOrEqual(2);
  });

  it("notifies the caller when the tool name is clicked", async () => {
    const onSelectTool = vi.fn();
    renderTable({ onSelectTool });

    await userEvent.click(screen.getByText("search_docs"));

    expect(onSelectTool).toHaveBeenCalledWith("search_docs");
  });

  it("renders a policy control for each direction, showing the row's current policies", () => {
    renderTable();

    expect(screen.getByText("untrusted")).toBeInTheDocument();
    expect(screen.getByText("trusted")).toBeInTheDocument();
    expect(screen.getAllByRole("combobox")).toHaveLength(2);
  });

  it("disables only the input policy control while that direction is saving", () => {
    renderTable({ savingInput: new Set(["search_docs"]) });

    const [input, output] = screen.getAllByRole("combobox");
    expect(input).toBeDisabled();
    expect(output).toBeEnabled();
  });

  it("disables only the output policy control while that direction is saving", () => {
    renderTable({ savingOutput: new Set(["search_docs"]) });

    const [input, output] = screen.getAllByRole("combobox");
    expect(input).toBeEnabled();
    expect(output).toBeDisabled();
  });
});
