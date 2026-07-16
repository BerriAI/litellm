import { render, screen } from "@testing-library/react";
import { vi } from "vitest";
import { flexRender, getCoreRowModel, useReactTable } from "@tanstack/react-table";
import { mcpHubColumns, MCPServerData } from "./mcp_hub_table_columns";

const SERVER_URL = "https://mcp.exa.ai/mcp";

const mockServer: MCPServerData = {
  server_id: "server-1",
  server_name: "exa_test",
  description: "Fast, intelligent web search and web crawling",
  url: SERVER_URL,
  transport: "http",
  auth_type: "none",
  created_at: "2026-01-01T00:00:00Z",
  created_by: "admin",
  updated_at: "2026-01-01T00:00:00Z",
  updated_by: "admin",
  teams: [],
  mcp_access_groups: [],
  allowed_tools: [],
  extra_headers: [],
  mcp_info: {},
  static_headers: {},
  status: "active",
  args: [],
  env: {},
};

function TestTable({ data }: { data: MCPServerData[] }) {
  const columns = mcpHubColumns(vi.fn(), vi.fn(), false);
  const table = useReactTable({ data, columns, getCoreRowModel: getCoreRowModel() });

  return (
    <table>
      <thead>
        {table.getHeaderGroups().map((hg) => (
          <tr key={hg.id}>
            {hg.headers.map((h) => (
              <th key={h.id}>{flexRender(h.column.columnDef.header, h.getContext())}</th>
            ))}
          </tr>
        ))}
      </thead>
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

describe("mcpHubColumns", () => {
  it("renders the server row", () => {
    render(<TestTable data={[mockServer]} />);
    expect(screen.getByText("exa_test")).toBeInTheDocument();
  });

  it("keeps the non-sensitive columns", () => {
    render(<TestTable data={[mockServer]} />);
    expect(screen.getByText("Server Name")).toBeInTheDocument();
    expect(screen.getByText("Transport")).toBeInTheDocument();
    expect(screen.getByText("Auth Type")).toBeInTheDocument();
  });

  it("does not expose a URL column header", () => {
    render(<TestTable data={[mockServer]} />);
    expect(screen.queryByText("URL")).not.toBeInTheDocument();
    expect(mcpHubColumns(vi.fn(), vi.fn(), false).some((c) => c.header === "URL")).toBe(false);
  });

  it("does not render the server url anywhere in the table", () => {
    render(<TestTable data={[mockServer]} />);
    expect(screen.queryByText(SERVER_URL)).not.toBeInTheDocument();
  });
});
