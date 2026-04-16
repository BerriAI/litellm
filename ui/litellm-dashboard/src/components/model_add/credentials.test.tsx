import { credentialCreateCall, CredentialItem } from "@/components/networking";
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
  credentialsKeys: { all: ["credentials"] },
}));

vi.mock("@/components/networking", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/components/networking")>();
  return {
    ...actual,
    credentialCreateCall: vi.fn(),
  };
});

// Mock the modal to expose a direct submit trigger so tests don't need to
// interact with Ant Design form internals and multi-step validation.
vi.mock("./AddCredentialModal", () => ({
  default: ({ open, onAddCredential }: any) =>
    open ? (
      <button
        data-testid="mock-add-credential-submit"
        onClick={() => onAddCredential({ credential_name: "test-cred", custom_llm_provider: "openai" })}
      >
        Submit Credential
      </button>
    ) : null,
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
      expect(screen.getByTestId("mock-add-credential-submit")).toBeInTheDocument();
    });
  });

  it("should invalidate shared credentials cache after adding a credential", async () => {
    const queryClient = createQueryClient();
    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries").mockResolvedValue();

    mockUseAuthorized.mockReturnValue({ accessToken: "test-token" });
    mockUseCredentials.mockReturnValue({ data: { credentials: [] } });
    (credentialCreateCall as ReturnType<typeof vi.fn>).mockResolvedValue({});

    render(
      <QueryClientProvider client={queryClient}>
        <CredentialsPanel uploadProps={DEFAULT_UPLOAD_PROPS} />
      </QueryClientProvider>,
    );

    // Open modal
    act(() => {
      fireEvent.click(screen.getByRole("button", { name: /add credential/i }));
    });

    // Trigger submission via the mocked modal's direct submit button
    await waitFor(() => screen.getByTestId("mock-add-credential-submit"));
    await act(async () => {
      fireEvent.click(screen.getByTestId("mock-add-credential-submit"));
    });

    await waitFor(() => {
      expect(invalidateSpy).toHaveBeenCalledWith(
        expect.objectContaining({ queryKey: ["credentials"] }),
      );
    });
  });
});
