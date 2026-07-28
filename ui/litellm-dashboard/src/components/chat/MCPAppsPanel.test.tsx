import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import MCPAppsPanel from "./MCPAppsPanel";
import { fetchMCPServers, listMCPTools } from "../networking";
import type { MCPServer } from "../mcp_tools/types";
import { setServerRootPath } from "@/lib/serverRootPath";

vi.mock("../networking", () => ({
  fetchMCPServers: vi.fn(),
  getMCPOAuthUserCredentialStatus: vi.fn(),
  listMCPTools: vi.fn(),
  deleteMCPOAuthUserCredential: vi.fn(),
}));

vi.mock("@/hooks/useUserMcpOAuthFlow", () => ({
  useUserMcpOAuthFlow: () => ({ startOAuthFlow: vi.fn(), status: "idle" }),
}));

const servers = [
  {
    server_id: "s-ext",
    server_name: "external_logo",
    auth_type: "none",
    mcp_info: { server_name: "external_logo", logo_url: "https://cdn.example.com/ext.png" },
  },
  {
    server_id: "s-local",
    server_name: "local_logo",
    auth_type: "none",
    mcp_info: { server_name: "local_logo", logo_url: "/ui/assets/logos/github.svg" },
  },
  {
    server_id: "s-none",
    server_name: "no_logo",
    auth_type: "none",
  },
] as MCPServer[];

const renderPanel = () =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <MCPAppsPanel accessToken="tok" selectedServers={[]} onChange={vi.fn()} />
    </QueryClientProvider>,
  );

describe("MCPAppsPanel logos", () => {
  afterEach(() => {
    setServerRootPath("/");
  });

  it("resolves backend logo_url values in the server grid", async () => {
    setServerRootPath("/litellm");
    vi.mocked(fetchMCPServers).mockResolvedValue(servers);
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });

    renderPanel();

    expect(await screen.findByText("external_logo")).toBeInTheDocument();
    expect(screen.getByAltText("external_logo logo").getAttribute("src")).toBe("https://cdn.example.com/ext.png");
    expect(screen.getByAltText("local_logo logo").getAttribute("src")).toBe("/litellm/ui/assets/logos/github.svg");
  });

  it("renders a colored letter avatar for servers without logo_url", async () => {
    vi.mocked(fetchMCPServers).mockResolvedValue(servers);
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });

    renderPanel();

    expect(await screen.findByText("no_logo")).toBeInTheDocument();
    expect(screen.queryByAltText("no_logo logo")).not.toBeInTheDocument();
    expect(screen.getByText("N")).toBeInTheDocument();
  });

  it("resolves the logo_url in the detail header", async () => {
    setServerRootPath("/litellm");
    vi.mocked(fetchMCPServers).mockResolvedValue(servers);
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });

    renderPanel();

    fireEvent.click(await screen.findByText("local_logo"));

    expect(await screen.findByRole("heading", { name: "local_logo" })).toBeInTheDocument();
    expect(screen.getByAltText("local_logo logo").getAttribute("src")).toBe("/litellm/ui/assets/logos/github.svg");
  });
});

const connectServers = [
  {
    server_id: "s-reach",
    server_name: "reachable_srv",
    auth_type: "none",
    connected_app_reachable: true,
  },
  {
    server_id: "s-unreach",
    server_name: "unreachable_srv",
    auth_type: "none",
    connected_app_reachable: false,
  },
] as MCPServer[];

const renderConnectPanel = (connectMode: boolean, selectedServers: string[] = []) =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <MCPAppsPanel accessToken="tok" selectedServers={selectedServers} onChange={vi.fn()} connectMode={connectMode} />
    </QueryClientProvider>,
  );

describe("MCPAppsPanel connected-app reachability (LIT-4861)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("requests the connected-app view and labels unreachable servers in connect mode", async () => {
    vi.mocked(fetchMCPServers).mockResolvedValue(connectServers);
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });

    renderConnectPanel(true, ["reachable_srv", "unreachable_srv"]);

    expect(await screen.findByText("unreachable_srv")).toBeInTheDocument();
    expect(vi.mocked(fetchMCPServers)).toHaveBeenCalledWith("tok", undefined, true);
    expect(screen.getByText("Not available to connected apps")).toBeInTheDocument();
    expect(screen.getByText("Connected (1)")).toBeInTheDocument();
    const toolCountFetchedIds = vi.mocked(listMCPTools).mock.calls.map((call) => call[1]);
    expect(toolCountFetchedIds).toContain("s-reach");
    expect(toolCountFetchedIds).not.toContain("s-unreach");
  });

  it("ignores the flag and skips no server outside connect mode", async () => {
    vi.mocked(fetchMCPServers).mockResolvedValue(connectServers);
    vi.mocked(listMCPTools).mockResolvedValue({ tools: [] });

    renderConnectPanel(false, ["reachable_srv", "unreachable_srv"]);

    expect(await screen.findByText("unreachable_srv")).toBeInTheDocument();
    expect(vi.mocked(fetchMCPServers)).toHaveBeenCalledWith("tok", undefined, false);
    expect(screen.queryByText("Not available to connected apps")).not.toBeInTheDocument();
    expect(screen.getByText("Connected (2)")).toBeInTheDocument();
    const toolCountFetchedIds = vi.mocked(listMCPTools).mock.calls.map((call) => call[1]);
    expect(toolCountFetchedIds).toContain("s-unreach");
  });
});
