import { CredentialItem } from "@/components/networking";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { UploadProps } from "antd/es/upload";
import { describe, expect, it, vi } from "vitest";
import CredentialsPanel from "./credentials";

const DEFAULT_UPLOAD_PROPS = {} as UploadProps;

const mockUseAuthorized = vi.fn();
const mockUseCredentials = vi.fn();

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => mockUseAuthorized(),
}));

vi.mock("@/app/(dashboard)/hooks/credentials/useCredentials", () => ({
  useCredentials: () => mockUseCredentials(),
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

describe("CredentialsPanel", () => {
  it("should render", () => {
    mockUseAuthorized.mockReturnValue({ accessToken: "test-token" });
    mockUseCredentials.mockReturnValue({
      data: { credentials: [] },
      refetch: vi.fn(),
    });

    render(
      <QueryClientProvider client={createQueryClient()}>
        <CredentialsPanel uploadProps={DEFAULT_UPLOAD_PROPS} />
      </QueryClientProvider>,
    );

    expect(screen.getByRole("button", { name: /add credential/i })).toBeInTheDocument();
  });

  it("should display provided credentials", () => {
    const credentials: CredentialItem[] = [
      {
        credential_name: "openai-key",
        credential_values: {},
        credential_info: { custom_llm_provider: "openai" },
      },
    ];

    mockUseAuthorized.mockReturnValue({ accessToken: "test-token" });
    mockUseCredentials.mockReturnValue({
      data: { credentials },
      refetch: vi.fn(),
    });

    render(
      <QueryClientProvider client={createQueryClient()}>
        <CredentialsPanel uploadProps={DEFAULT_UPLOAD_PROPS} />
      </QueryClientProvider>,
    );

    expect(screen.getByText("openai-key")).toBeInTheDocument();
  });

  it("should display empty state when no credentials are provided", () => {
    mockUseAuthorized.mockReturnValue({ accessToken: "test-token" });
    mockUseCredentials.mockReturnValue({
      data: { credentials: [] },
      refetch: vi.fn(),
    });

    render(
      <QueryClientProvider client={createQueryClient()}>
        <CredentialsPanel uploadProps={DEFAULT_UPLOAD_PROPS} />
      </QueryClientProvider>,
    );

    expect(screen.getByText("No credentials configured")).toBeInTheDocument();
  });

  it("should open add modal when add button is clicked", async () => {
    mockUseAuthorized.mockReturnValue({ accessToken: "test-token" });
    mockUseCredentials.mockReturnValue({
      data: { credentials: [] },
      refetch: vi.fn(),
    });

    render(
      <QueryClientProvider client={createQueryClient()}>
        <CredentialsPanel uploadProps={DEFAULT_UPLOAD_PROPS} />
      </QueryClientProvider>,
    );

    const addButton = screen.getByRole("button", { name: /add credential/i });

    act(() => {
      fireEvent.click(addButton);
    });

    await waitFor(() => {
      expect(screen.getByText("Add New Credential")).toBeInTheDocument();
    });
  });
});
