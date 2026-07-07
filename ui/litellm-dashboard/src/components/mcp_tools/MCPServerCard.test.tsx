import React from "react";
import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import MCPServerCard from "./MCPServerCard";
import type { MCPServer } from "./types";

const baseServer: MCPServer = {
  server_id: "srv-1",
  server_name: "demo_server",
  alias: "demo_server",
  transport: "http",
  url: "https://example.com/mcp",
  auth_type: "oauth2",
} as MCPServer;

function renderCard(overrides: Partial<MCPServer>) {
  render(<MCPServerCard server={{ ...baseServer, ...overrides } as MCPServer} onClick={vi.fn()} />);
}

describe("MCPServerCard OAuth flow indicator", () => {
  it("shows the 'OAuth flow not set' badge for an oauth2 server with no oauth2_flow", () => {
    renderCard({ auth_type: "oauth2", oauth2_flow: null });
    expect(screen.getByText("OAuth flow not set")).toBeInTheDocument();
  });

  it("does not show the badge once oauth2_flow is set (client_credentials)", () => {
    renderCard({ auth_type: "oauth2", oauth2_flow: "client_credentials" });
    expect(screen.queryByText("OAuth flow not set")).not.toBeInTheDocument();
  });

  it("does not show the badge once oauth2_flow is set (authorization_code)", () => {
    renderCard({ auth_type: "oauth2", oauth2_flow: "authorization_code" });
    expect(screen.queryByText("OAuth flow not set")).not.toBeInTheDocument();
  });

  it("does not show the badge for a non-oauth2 server", () => {
    renderCard({ auth_type: "api_key", oauth2_flow: null });
    expect(screen.queryByText("OAuth flow not set")).not.toBeInTheDocument();
  });

  it("does not show the badge for a delegate (PKCE passthrough) server", () => {
    renderCard({ auth_type: "oauth2", oauth2_flow: null, delegate_auth_to_upstream: true });
    expect(screen.queryByText("OAuth flow not set")).not.toBeInTheDocument();
  });
});
