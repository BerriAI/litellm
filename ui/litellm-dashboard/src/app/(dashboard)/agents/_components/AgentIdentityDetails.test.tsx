import { screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders, testQueryClient } from "../../../../../tests/test-utils";
import { apiClient } from "@/components/networking";
import { AgentIdentityDetails } from "./AgentIdentityDetails";

vi.mock("@/components/networking", () => ({ apiClient: { get: vi.fn() } }));

const identity = {
  provider: "microsoft_entra",
  tenant_id: "11111111-1111-4111-8111-111111111111",
  client_id: "22222222-2222-4222-8222-222222222222",
  provisioning_source_id: "directory",
};

const status = {
  enabled: true,
  execution_mode: "autonomous",
  directory_active: true,
  directory_access_group_ids: ["read"],
  last_authenticated_at: "2026-09-24T12:00:00Z",
};

describe("agent identity evidence", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    testQueryClient.clear();
  });

  it("shows persisted native identity evidence and links to the current logs route", async () => {
    vi.mocked(apiClient.get).mockResolvedValue(status);
    renderWithProviders(<AgentIdentityDetails agentId="native" identity={identity} accessToken="admin" isAdmin />);
    expect(await screen.findByText(/Last authenticated identity match:/)).toBeInTheDocument();
    expect(screen.getByText(/Entra Parent Identity ID:/)).toBeInTheDocument();
    expect(screen.getByText("Directory status: Active")).toBeInTheDocument();
    expect(screen.getByText("Mapped access groups: 1")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "View request logs" })).toHaveAttribute("href", "/ui/logs/");
    expect(apiClient.get).toHaveBeenCalledWith("/v1/agents/native/identity", { accessToken: "admin" });
  });

  it("does not present unavailable directory status as active", () => {
    vi.mocked(apiClient.get).mockReturnValue(new Promise(() => {}));
    renderWithProviders(<AgentIdentityDetails agentId="native" identity={identity} accessToken="admin" isAdmin />);
    expect(screen.getByText("Directory status: Unavailable")).toBeInTheDocument();
    expect(screen.queryByText("Directory status: Active")).not.toBeInTheDocument();
  });

  it("does not request or show administrator identity evidence to ordinary users", () => {
    renderWithProviders(
      <AgentIdentityDetails agentId="native" identity={identity} accessToken="user" isAdmin={false} />,
    );
    expect(screen.queryByRole("region", { name: "Agent Identity" })).not.toBeInTheDocument();
    expect(apiClient.get).not.toHaveBeenCalled();
  });
});
