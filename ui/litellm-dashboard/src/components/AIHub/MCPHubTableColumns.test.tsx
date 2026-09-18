import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { DataTable } from "@/components/shared/DataTable";
import { getMCPHubTableColumns, MCPServerData } from "./MCPHubTableColumns";

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

function renderTable(onServerClick = vi.fn()) {
  render(
    <DataTable
      data={[mockServer]}
      columns={getMCPHubTableColumns({ onServerClick })}
      getRowId={(server) => server.server_id}
      sortingMode="client"
      size="compact"
    />,
  );
  return onServerClick;
}

describe("getMCPHubTableColumns", () => {
  it("renders the server row", () => {
    renderTable();
    expect(screen.getByText("exa_test")).toBeInTheDocument();
  });

  it("keeps the non-sensitive columns", () => {
    renderTable();
    expect(screen.getByText("Server Name")).toBeInTheDocument();
    expect(screen.getByText("Transport")).toBeInTheDocument();
    expect(screen.getByText("Auth Type")).toBeInTheDocument();
  });

  it("does not expose a URL column", () => {
    renderTable();
    expect(screen.queryByText("URL")).not.toBeInTheDocument();
    const columns = getMCPHubTableColumns({ onServerClick: vi.fn() });
    expect(columns.some((c) => c.header === "URL" || c.meta?.title === "URL")).toBe(false);
  });

  it("does not render the server url anywhere in the table", () => {
    renderTable();
    expect(screen.queryByText(SERVER_URL)).not.toBeInTheDocument();
  });

  it("opens the server details when the name is clicked", async () => {
    const user = userEvent.setup();
    const onServerClick = renderTable();
    await user.click(screen.getByRole("button", { name: "exa_test" }));
    expect(onServerClick).toHaveBeenCalledWith(mockServer);
  });

  it("opens the server details from the actions menu", async () => {
    const user = userEvent.setup();
    const onServerClick = renderTable();
    await user.click(screen.getByTestId("mcp-hub-actions-server-1"));
    await user.click(await screen.findByTestId("mcp-hub-action-details"));
    expect(onServerClick).toHaveBeenCalledWith(mockServer);
  });

  it("copies the server name from the actions menu", async () => {
    const user = userEvent.setup();
    renderTable();
    await user.click(screen.getByTestId("mcp-hub-actions-server-1"));
    await user.click(await screen.findByTestId("mcp-hub-action-copy"));
    expect(await window.navigator.clipboard.readText()).toBe("exa_test");
  });
});
