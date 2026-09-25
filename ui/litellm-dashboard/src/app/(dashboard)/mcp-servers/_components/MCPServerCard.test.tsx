import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, it, expect, vi, afterEach } from "vitest";
import MCPServerCard from "./MCPServerCard";
import type { MCPServer } from "@/components/mcp_tools/types";
import { setServerRootPath } from "@/lib/serverRootPath";

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

describe("MCPServerCard logo", () => {
  afterEach(() => {
    setServerRootPath("/");
  });

  it("passes an external logo_url through untouched", () => {
    renderCard({ mcp_info: { server_name: "demo_server", logo_url: "https://cdn.example.com/logo.png" } });
    expect(screen.getByAltText("demo_server logo")).toHaveAttribute("src", "https://cdn.example.com/logo.png");
  });

  it("prefixes a stored asset path with the server root path under a non-root mount", () => {
    setServerRootPath("/litellm");
    renderCard({ mcp_info: { server_name: "demo_server", logo_url: "/ui/assets/logos/github.svg" } });
    expect(screen.getByAltText("demo_server logo")).toHaveAttribute("src", "/litellm/ui/assets/logos/github.svg");
  });

  it("renders a letter avatar when no logo_url is set", () => {
    renderCard({ mcp_info: { server_name: "demo_server" } });
    expect(screen.queryByAltText("demo_server logo")).not.toBeInTheDocument();
    expect(screen.getByText("DE")).toBeInTheDocument();
  });
});

describe("MCPServerCard per-user credentials", () => {
  const renderUserFields = (props: { missingUserFields?: string[]; hasUserFields?: boolean }) => {
    const onOpenFillFields = vi.fn();
    const onClick = vi.fn();
    render(<MCPServerCard server={baseServer} onClick={onClick} onOpenFillFields={onOpenFillFields} {...props} />);
    return { onOpenFillFields, onClick };
  };

  it("offers Set while a field is missing", () => {
    const { onOpenFillFields, onClick } = renderUserFields({ missingUserFields: ["USER_TOKEN"], hasUserFields: true });
    expect(screen.getByText("1 user field missing")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Update" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Set" }));
    expect(onOpenFillFields).toHaveBeenCalledTimes(1);
    expect(onClick).not.toHaveBeenCalled();
  });

  it("keeps an Update entry point once every field is set", () => {
    const { onOpenFillFields, onClick } = renderUserFields({ missingUserFields: [], hasUserFields: true });
    expect(screen.getByText("Per-user credentials")).toBeInTheDocument();
    expect(screen.getByText("Set")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Set" })).not.toBeInTheDocument();
    expect(screen.queryByText(/user field/)).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Update" }));
    expect(onOpenFillFields).toHaveBeenCalledTimes(1);
    expect(onClick).not.toHaveBeenCalled();
  });

  it("keeps Enter on the Update button away from the card's open handler", () => {
    const { onClick } = renderUserFields({ missingUserFields: [], hasUserFields: true });
    const update = screen.getByRole("button", { name: "Update" });
    expect(fireEvent.keyDown(update, { key: "Enter" }), "default activation must survive").toBe(true);
    expect(onClick).not.toHaveBeenCalled();
    fireEvent.keyDown(screen.getAllByRole("button")[0], { key: "Enter" });
    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it("renders no credential row for a server without per-user fields", () => {
    renderUserFields({ missingUserFields: [], hasUserFields: false });
    expect(screen.queryByText("Per-user credentials")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Update" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Set" })).not.toBeInTheDocument();
  });
});

describe("MCPServerCard network access", () => {
  it("shows effective network access without a hub listing badge", () => {
    renderCard({
      available_on_public_internet: false,
      mcp_info: { server_name: "demo_server", is_public: true, is_public_explicit: true },
    });

    expect(screen.getByText("All Networks")).toBeInTheDocument();
    expect(screen.queryByText(/^Hub:/)).not.toBeInTheDocument();
  });
});
