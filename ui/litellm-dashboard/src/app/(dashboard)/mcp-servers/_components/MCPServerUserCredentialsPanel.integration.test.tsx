import React from "react";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MCPServerUserCredentialsPanel } from "./MCPServerUserCredentialsPanel";
import * as networking from "@/components/networking";
import type { MCPServerUserCredentialListItem } from "@/components/mcp_tools/types";

vi.mock("@/components/networking", () => ({
  fetchMCPServerUserCredentials: vi.fn(),
  revokeMCPServerUserCredential: vi.fn(),
}));

const ITEMS: MCPServerUserCredentialListItem[] = [
  {
    user_id: "alice",
    credential_type: "oauth2",
    expires_at: "2026-12-31T00:00:00+00:00",
    connected_at: "2026-01-01T00:00:00+00:00",
    updated_at: "2026-01-01T00:00:00+00:00",
  },
  {
    user_id: "carol",
    credential_type: "byok",
    expires_at: null,
    connected_at: null,
    updated_at: "2026-02-01T00:00:00+00:00",
  },
];

const renderPanel = ({ canRevoke = false }: { canRevoke?: boolean } = {}) => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <MCPServerUserCredentialsPanel serverId="srv-1" accessToken="token" canRevoke={canRevoke} />
    </QueryClientProvider>,
  );
};

describe("MCPServerUserCredentialsPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("lists each user's credential type without a revoke control for a read-only admin", async () => {
    vi.mocked(networking.fetchMCPServerUserCredentials).mockResolvedValue(ITEMS);
    renderPanel({ canRevoke: false });

    const table = await screen.findByRole("region", { name: "Stored user credentials" });
    expect(within(table).getByRole("row", { name: /alice/ })).toHaveTextContent("OAuth2");
    expect(within(table).getByRole("row", { name: /carol/ })).toHaveTextContent("BYOK API key");
    expect(screen.queryByRole("button", { name: /^Revoke credential/ })).not.toBeInTheDocument();
    expect(networking.fetchMCPServerUserCredentials).toHaveBeenCalledWith("token", "srv-1");
  });

  it("revokes the selected user's credential through the route for its type and refetches", async () => {
    const user = userEvent.setup();
    vi.mocked(networking.fetchMCPServerUserCredentials).mockResolvedValueOnce(ITEMS).mockResolvedValueOnce([ITEMS[1]]);
    vi.mocked(networking.revokeMCPServerUserCredential).mockResolvedValue(undefined);
    renderPanel({ canRevoke: true });

    await user.click(await screen.findByRole("button", { name: "Revoke credential for user alice" }));
    expect(networking.revokeMCPServerUserCredential).not.toHaveBeenCalled();
    const dialog = await screen.findByRole("alertdialog");
    expect(dialog).toHaveTextContent("OAuth2 credential stored for user alice");
    await user.click(within(dialog).getByRole("button", { name: "Revoke" }));

    expect(await screen.findByText(/OAuth2 credential for user alice was deleted/)).toBeInTheDocument();
    expect(networking.revokeMCPServerUserCredential).toHaveBeenCalledWith("token", "srv-1", "alice", "oauth2");
    const table = await screen.findByRole("region", { name: "Stored user credentials" });
    expect(within(table).queryByRole("row", { name: /alice/ })).not.toBeInTheDocument();
    expect(within(table).getByRole("row", { name: /carol/ })).toBeInTheDocument();
  });

  it("shows the API error when a revoke is refused and keeps the list", async () => {
    const user = userEvent.setup();
    vi.mocked(networking.fetchMCPServerUserCredentials).mockResolvedValue(ITEMS);
    vi.mocked(networking.revokeMCPServerUserCredential).mockRejectedValue(
      new Error("Proxy admin access required to revoke another user's MCP credential."),
    );
    renderPanel({ canRevoke: true });

    await user.click(await screen.findByRole("button", { name: "Revoke credential for user carol" }));
    await user.click(within(await screen.findByRole("alertdialog")).getByRole("button", { name: "Revoke" }));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Could not revoke credential");
    expect(alert).toHaveTextContent("Proxy admin access required to revoke another user's MCP credential.");
    expect(networking.revokeMCPServerUserCredential).toHaveBeenCalledWith("token", "srv-1", "carol", "byok");
    expect(screen.getByRole("region", { name: "Stored user credentials" })).toBeInTheDocument();
  });

  it("shows the API error when the list cannot be loaded", async () => {
    vi.mocked(networking.fetchMCPServerUserCredentials).mockRejectedValue(new Error("Admin access required"));
    renderPanel();

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Could not load user credentials");
    expect(alert).toHaveTextContent("Admin access required");
  });
});
