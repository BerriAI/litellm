import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { UploadProps } from "antd/es/upload";
import { describe, expect, it, vi } from "vitest";

import { CredentialItem } from "@/components/networking";

import CredentialsPanel from "./CredentialsPanel";

const DEFAULT_UPLOAD_PROPS = {} as UploadProps;

const mockUseAuthorized = vi.fn();
const mockUseCredentials = vi.fn();

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => mockUseAuthorized(),
}));

vi.mock("@/app/(dashboard)/hooks/credentials/useCredentials", () => ({
  useCredentials: () => mockUseCredentials(),
}));

const credentials: CredentialItem[] = [
  {
    credential_name: "openai-key",
    credential_values: {},
    credential_info: { custom_llm_provider: "openai" },
  },
];

const createQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
      },
    },
  });

const renderPanel = () =>
  render(
    <QueryClientProvider client={createQueryClient()}>
      <CredentialsPanel uploadProps={DEFAULT_UPLOAD_PROPS} />
    </QueryClientProvider>,
  );

describe("CredentialsPanel", () => {
  it("renders the Add Credential button for an admin", () => {
    mockUseAuthorized.mockReturnValue({ accessToken: "test-token", userRole: "Admin" });
    mockUseCredentials.mockReturnValue({ data: { credentials: [] }, isLoading: false, refetch: vi.fn() });

    renderPanel();

    expect(screen.getByRole("button", { name: /add credential/i })).toBeInTheDocument();
  });

  it("displays the credential rows", () => {
    mockUseAuthorized.mockReturnValue({ accessToken: "test-token", userRole: "Admin" });
    mockUseCredentials.mockReturnValue({ data: { credentials }, isLoading: false, refetch: vi.fn() });

    renderPanel();

    expect(screen.getByText("openai-key")).toBeInTheDocument();
  });

  it("shows the empty state when there are no credentials", () => {
    mockUseAuthorized.mockReturnValue({ accessToken: "test-token", userRole: "Admin" });
    mockUseCredentials.mockReturnValue({ data: { credentials: [] }, isLoading: false, refetch: vi.fn() });

    renderPanel();

    expect(screen.getByText("No credentials configured")).toBeInTheDocument();
  });

  it("shows the loading skeleton instead of the empty state while credentials load", () => {
    mockUseAuthorized.mockReturnValue({ accessToken: "test-token", userRole: "Admin" });
    mockUseCredentials.mockReturnValue({ data: undefined, isLoading: true, refetch: vi.fn() });

    renderPanel();

    // isLoading must reach the table: the empty state must not render mid-load.
    expect(screen.queryByText("No credentials configured")).not.toBeInTheDocument();
  });

  it("opens the add modal when the add button is clicked", async () => {
    mockUseAuthorized.mockReturnValue({ accessToken: "test-token", userRole: "Admin" });
    mockUseCredentials.mockReturnValue({ data: { credentials: [] }, isLoading: false, refetch: vi.fn() });

    renderPanel();

    act(() => {
      fireEvent.click(screen.getByRole("button", { name: /add credential/i }));
    });

    await waitFor(() => {
      expect(screen.getByText("Add New Credential")).toBeInTheDocument();
    });
  });

  describe("Admin Viewer write-action gating", () => {
    // Admin Viewer can VIEW credentials but must not add / edit / delete them.
    it("hides the Add Credential button but still lists credentials", () => {
      mockUseAuthorized.mockReturnValue({ accessToken: "test-token", userRole: "Admin Viewer" });
      mockUseCredentials.mockReturnValue({ data: { credentials }, isLoading: false, refetch: vi.fn() });

      renderPanel();

      expect(screen.getByText("openai-key")).toBeInTheDocument();
      expect(screen.queryByRole("button", { name: /add credential/i })).not.toBeInTheDocument();
    });

    it("does not render the per-row actions menu for Admin Viewer", () => {
      mockUseAuthorized.mockReturnValue({ accessToken: "test-token", userRole: "Admin Viewer" });
      mockUseCredentials.mockReturnValue({ data: { credentials }, isLoading: false, refetch: vi.fn() });

      renderPanel();

      expect(screen.queryByTestId("credential-actions-openai-key")).not.toBeInTheDocument();
    });
  });
});
