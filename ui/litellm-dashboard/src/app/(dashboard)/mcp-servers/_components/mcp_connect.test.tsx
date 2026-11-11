import React from "react";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi } from "vitest";
import MCPConnect from "./mcp_connect";

vi.mock("@/components/networking", () => ({
  getProxyBaseUrl: vi.fn().mockReturnValue("https://backend.example.com"),
  getGatewayBaseUrl: vi.fn().mockReturnValue("https://gateway.example.com"),
}));

const activePanel = () => screen.getByRole("tabpanel");

describe("MCPConnect routing", () => {
  it("uses the gateway for Responses and the backend for MCP server URLs", async () => {
    const user = userEvent.setup();
    render(<MCPConnect />);

    expect(within(activePanel()).getByText(/"server_url": "https:\/\/backend\.example\.com\/mcp"/)).toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: "LiteLLM Proxy" }));

    expect(within(activePanel()).getByText("https://backend.example.com/mcp")).toBeInTheDocument();
    expect(
      within(activePanel()).getByText(/curl --location 'https:\/\/gateway\.example\.com\/v1\/responses'/),
    ).toBeInTheDocument();
  });
});

describe("MCPConnect (tab mount contract)", () => {
  it("keeps the x-mcp-servers header toggle on after switching tabs away and back", async () => {
    render(<MCPConnect />);

    await userEvent.click(screen.getByRole("tab", { name: "LiteLLM Proxy" }));
    expect(within(activePanel()).queryByText(/"x-mcp-servers":/)).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole("switch"));
    expect(within(activePanel()).getByText(/"x-mcp-servers": "Zapier_MCP,dev-group"/)).toBeInTheDocument();

    await userEvent.click(screen.getByRole("tab", { name: "Cursor" }));
    await userEvent.click(screen.getByRole("tab", { name: "LiteLLM Proxy" }));

    expect(screen.getByRole("switch")).toBeChecked();
    expect(within(activePanel()).getByText(/"x-mcp-servers": "Zapier_MCP,dev-group"/)).toBeInTheDocument();
  });
});
