import React from "react";
import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi, afterEach } from "vitest";
import MCPConnectPicker from "./MCPConnectPicker";
import { fetchMCPServers } from "../networking";
import type { MCPServer } from "../mcp_tools/types";
import { setServerRootPath } from "@/lib/serverRootPath";

vi.mock("../networking", () => ({
  fetchMCPServers: vi.fn(),
  listMCPTools: vi.fn(),
}));

const servers = [
  {
    server_id: "s-ext",
    server_name: "external_logo",
    mcp_info: { server_name: "external_logo", logo_url: "https://cdn.example.com/ext.png" },
  },
  {
    server_id: "s-local",
    server_name: "local_logo",
    mcp_info: { server_name: "local_logo", logo_url: "/ui/assets/logos/github.svg" },
  },
  {
    server_id: "s-none",
    server_name: "no_logo",
  },
] as MCPServer[];

describe("MCPConnectPicker logos", () => {
  afterEach(() => {
    setServerRootPath("/");
  });

  it("resolves backend logo_url values through the Logo component", async () => {
    setServerRootPath("/litellm");
    vi.mocked(fetchMCPServers).mockResolvedValue(servers);

    render(<MCPConnectPicker accessToken="tok" selectedServers={[]} onChange={vi.fn()} />);

    expect(await screen.findByText("external_logo")).toBeInTheDocument();
    expect(screen.getByAltText("external_logo logo").getAttribute("src")).toBe("https://cdn.example.com/ext.png");
    expect(screen.getByAltText("local_logo logo").getAttribute("src")).toBe("/litellm/ui/assets/logos/github.svg");
  });

  it("renders no logo at all for servers without logo_url", async () => {
    vi.mocked(fetchMCPServers).mockResolvedValue(servers);

    render(<MCPConnectPicker accessToken="tok" selectedServers={[]} onChange={vi.fn()} />);

    expect(await screen.findByText("no_logo")).toBeInTheDocument();
    expect(screen.queryByAltText("no_logo logo")).not.toBeInTheDocument();
  });
});
