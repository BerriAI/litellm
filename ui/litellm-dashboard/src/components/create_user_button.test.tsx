import React from "react";
import { render } from "@testing-library/react";
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
});
