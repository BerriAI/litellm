import React from "react";
import { render, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import Createuser from "./create_user_button";

vi.mock("./networking", () => ({
  userCreateCall: vi.fn(),
  modelAvailableCall: vi.fn().mockResolvedValue({ data: [] }),
  invitationCreateCall: vi.fn(),
  getProxyUISettings: vi.fn().mockResolvedValue({
    PROXY_BASE_URL: null,
    PROXY_LOGOUT_URL: null,
    DEFAULT_TEAM_DISABLED: false,
    SSO_ENABLED: false,
  }),
  getProxyBaseUrl: vi.fn().mockReturnValue("http://localhost"),
}));

const createQueryClient = () =>
  new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });

describe("Create User Button", () => {
  it("should render the create user button", () => {
    const qc = createQueryClient();
    const { getByText } = render(
      <QueryClientProvider client={qc}>
        <Createuser userID="123" accessToken="123" teams={[]} possibleUIRoles={{}} isEmbedded />
      </QueryClientProvider>,
    );
    expect(getByText("Create User")).toBeInTheDocument();
  });

  it("should render send invite email toggle in modal with default off", async () => {
    const qc = createQueryClient();
    const { getByText, getByRole } = render(
      <QueryClientProvider client={qc}>
        <Createuser userID="123" accessToken="123" teams={[]} possibleUIRoles={{}} isEmbedded={false} />
      </QueryClientProvider>,
    );

    // Click the "+ Invite User" button to open modal
    await waitFor(() => {
      expect(getByText("+ Invite User")).toBeInTheDocument();
    });
    const inviteButton = getByText("+ Invite User");
    fireEvent.click(inviteButton);

    // Wait for modal to open and check for "Send Invite Email" label
    await waitFor(() => {
      expect(getByText("Send Invite Email")).toBeInTheDocument();
    });

    // Check that the switch toggle exists and is OFF by default
    const switchToggle = getByRole("switch");
    expect(switchToggle).toBeInTheDocument();
    expect(switchToggle).not.toBeChecked();
  });
});
