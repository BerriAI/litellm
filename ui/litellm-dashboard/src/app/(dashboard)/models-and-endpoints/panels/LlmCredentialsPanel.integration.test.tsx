import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import LlmCredentialsPanel from "./LlmCredentialsPanel";

const networking = vi.hoisted(() => ({
  credentialListCall: vi.fn(),
  userConnectionsListCall: vi.fn(),
  userConnectionDeleteCall: vi.fn(),
}));

vi.mock("@/components/networking", async () => ({
  ...(await vi.importActual<object>("@/components/networking")),
  ...networking,
}));

const mockUseAuthorized = vi.fn();
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: () => mockUseAuthorized() }));

const session = (userRole: string) => ({ accessToken: "at", userRole, userId: "u1", isViewOnly: false });

const renderPanel = () =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } })}>
      <LlmCredentialsPanel />
    </QueryClientProvider>,
  );

describe("LlmCredentialsPanel", () => {
  beforeEach(() => {
    networking.credentialListCall.mockReset().mockResolvedValue({
      credentials: [
        { credential_name: "shared-openai", credential_values: {}, credential_info: { custom_llm_provider: "openai" } },
      ],
    });
    networking.userConnectionsListCall.mockReset().mockResolvedValue({
      connections: [
        {
          credential_name: "copilot-per-user",
          provider: "github_copilot",
          connected: false,
          github_login: null,
          connected_at: null,
        },
        {
          credential_name: "copilot-team",
          provider: "github_copilot",
          connected: true,
          github_login: "octocat",
          connected_at: "2026-10-07T00:00:00Z",
        },
      ],
    });
    networking.userConnectionDeleteCall.mockReset().mockResolvedValue(undefined);
  });

  it("shows an internal user only their connections and never fetches the shared credential list", async () => {
    mockUseAuthorized.mockReturnValue(session("Internal User"));
    renderPanel();

    expect(await screen.findByText("copilot-per-user")).toBeInTheDocument();
    expect(screen.getByText("Your connections")).toBeInTheDocument();
    expect(screen.getByText("Connected as @octocat")).toBeInTheDocument();
    expect(screen.queryByText("shared-openai")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Add Credential/ })).not.toBeInTheDocument();
    expect(networking.credentialListCall).not.toHaveBeenCalled();
  });

  it("shows an admin the credential list and their connections", async () => {
    mockUseAuthorized.mockReturnValue(session("Admin"));
    renderPanel();

    expect(await screen.findByText("shared-openai")).toBeInTheDocument();
    expect(await screen.findByText("copilot-per-user")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Add Credential/ })).toBeInTheDocument();
  });

  it("disconnects the selected connection and refreshes the list", async () => {
    const user = userEvent.setup();
    mockUseAuthorized.mockReturnValue(session("Internal User"));
    renderPanel();

    const row = await screen.findByRole("row", { name: /copilot-team/ });
    await user.click(within(row).getByRole("button", { name: "Disconnect" }));

    expect(networking.userConnectionDeleteCall).toHaveBeenCalledWith("at", "copilot-team");
    await waitFor(() => expect(networking.userConnectionsListCall).toHaveBeenCalledTimes(2));
  });
});
