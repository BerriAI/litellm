import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { apiClient } from "@/components/networking";
import TeamAgentsTab from "./TeamAgentsTab";

vi.mock("@/components/networking", () => ({ apiClient: { get: vi.fn(), put: vi.fn() } }));
const tenant = "11111111-1111-4111-8111-111111111111";
const client = "22222222-2222-4222-8222-222222222222";
const identity = { provider: "microsoft_entra", tenant_id: tenant, client_id: client };
const research = {
  agent_id: "research-id",
  agent_name: "Research",
  litellm_params: { team_id: "research-team", identity },
};
const publisher = { agent_id: "publisher-id", agent_name: "Publisher", litellm_params: { team_id: "publisher-team" } };
const unassigned = { agent_id: "new-id", agent_name: "New Agent", litellm_params: { team_id: null, identity } };

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(apiClient.get).mockResolvedValue([research, publisher, unassigned]);
  vi.mocked(apiClient.put).mockResolvedValue({});
});

const renderTeam = (canManage = true) => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <TeamAgentsTab teamId="research-team" accessToken="test-token" canManage={canManage} />
    </QueryClientProvider>,
  );
};

describe("team agent membership", () => {
  it("shows only assigned members with their canonical agent and Entra identifiers", async () => {
    renderTeam();
    const member = await screen.findByRole("article", { name: "Research" });
    expect(within(member).getByText("research-id")).toBeVisible();
    expect(within(member).getByText(tenant)).toBeVisible();
    expect(within(member).getByText(client)).toBeVisible();
    expect(screen.queryByRole("article", { name: "Publisher" })).not.toBeInTheDocument();
  });

  it("adds an unassigned registered agent and refreshes the membership list", async () => {
    const user = userEvent.setup();
    renderTeam();
    await screen.findByRole("article", { name: "Research" });
    fireEvent.change(screen.getByRole("textbox", { name: "Search unassigned agents" }), {
      target: { value: "new-id" },
    });
    await user.click(screen.getByRole("combobox", { name: "Agent to add" }));
    expect(screen.queryByRole("option", { name: /Publisher/ })).not.toBeInTheDocument();
    await user.click(screen.getByRole("option", { name: "New Agent (new-id)" }));
    expect(screen.getByText(/Microsoft Entra ID · Tenant:/)).toHaveTextContent(client);
    vi.mocked(apiClient.get).mockResolvedValue([
      research,
      publisher,
      { ...unassigned, litellm_params: { team_id: "research-team", identity } },
    ]);
    await user.click(screen.getByRole("button", { name: "Add agent to team" }));
    expect(await screen.findByRole("article", { name: "New Agent" })).toBeVisible();
    expect(apiClient.put).toHaveBeenCalledWith("/v1/agents/new-id/team", {
      accessToken: "test-token",
      body: { team_id: "research-team" },
    });
  });

  it("removes membership without changing the allowed-agent call list", async () => {
    const user = userEvent.setup();
    renderTeam();
    await screen.findByRole("article", { name: "Research" });
    vi.mocked(apiClient.get).mockResolvedValue([publisher, unassigned]);
    await user.click(screen.getByRole("button", { name: "Remove Research from team" }));
    expect(await screen.findByText("No agents assigned to this team")).toBeVisible();
    expect(apiClient.put).toHaveBeenCalledWith("/v1/agents/research-id/team", {
      accessToken: "test-token",
      body: { team_id: null },
    });
    expect(screen.getByRole("status")).toHaveTextContent("JWT requests are denied");
  });

  it("keeps membership visible when saving fails", async () => {
    const user = userEvent.setup();
    vi.mocked(apiClient.put).mockRejectedValue(new Error("failed"));
    renderTeam();
    await user.click(await screen.findByRole("button", { name: "Remove Research from team" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not update agent membership");
    expect(screen.getByRole("article", { name: "Research" })).toBeVisible();
  });

  it("reports loading errors and allows retry", async () => {
    vi.mocked(apiClient.get).mockRejectedValueOnce(new Error("failed"));
    renderTeam();
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not load agent memberships");
    fireEvent.click(screen.getByRole("button", { name: "Refresh agents" }));
    expect(await screen.findByRole("article", { name: "Research" })).toBeVisible();
    await waitFor(() => expect(screen.queryByRole("alert")).not.toBeInTheDocument());
  });

  it("requires a proxy administrator to manage assignments", () => {
    renderTeam(false);
    expect(screen.getByText("A proxy administrator manages agent team assignments.")).toBeVisible();
    expect(apiClient.get).not.toHaveBeenCalled();
    expect(screen.queryByRole("button", { name: "Add agent to team" })).not.toBeInTheDocument();
  });
});
