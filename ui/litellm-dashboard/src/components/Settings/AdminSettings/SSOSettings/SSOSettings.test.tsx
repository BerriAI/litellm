import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import SSOSettings from "./SSOSettings";

const mockUseSSOSettings = vi.fn();

// Mock the useSSOSettings hook
vi.mock("@/app/(dashboard)/hooks/sso/useSSOSettings", () => ({
  useSSOSettings: () => mockUseSSOSettings(),
}));

const createQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
      },
    },
  });

const renderSSOSettings = () => {
  const queryClient = createQueryClient();

  return render(
    <QueryClientProvider client={queryClient}>
      <SSOSettings />
    </QueryClientProvider>,
  );
};

const googleConfiguredValues = {
  google_client_id: "google-client-id",
  google_client_secret: "google-client-secret",
  microsoft_client_id: null,
  microsoft_client_secret: null,
  microsoft_tenant: null,
  generic_client_id: null,
  generic_client_secret: null,
  generic_authorization_endpoint: null,
  generic_token_endpoint: null,
  generic_userinfo_endpoint: null,
  proxy_base_url: null,
  user_email: null,
  ui_access_mode: null,
  role_mappings: null,
  team_mappings: null,
};

describe("SSOSettings", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseSSOSettings.mockReturnValue({
      data: null,
      isLoading: false,
      refetch: vi.fn(),
    });
  });

  it("should render", () => {
    renderSSOSettings();

    expect(screen.getByText("SSO Configuration")).toBeInTheDocument();
    expect(screen.getByText("Manage Single Sign-On authentication settings")).toBeInTheDocument();
  });

  it("shows the local google logo asset for a google-configured settings payload", () => {
    mockUseSSOSettings.mockReturnValue({
      data: { values: googleConfiguredValues },
      isLoading: false,
      refetch: vi.fn(),
    });

    renderSSOSettings();

    const logo = screen.getByAltText("Google SSO logo");
    expect(logo).toHaveAttribute("src", expect.stringContaining("google.svg"));
  });
});
