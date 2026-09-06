import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { CredentialItem, credentialUpdateCall } from "@/components/networking";

import CredentialsPanel from "./CredentialsPanel";

const mockUseAuthorized = vi.fn();
const mockUseCredentials = vi.fn();

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => mockUseAuthorized(),
}));

vi.mock("@/app/(dashboard)/hooks/credentials/useCredentials", () => ({
  useCredentials: () => mockUseCredentials(),
}));

vi.mock("@/components/networking", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/components/networking")>();
  return {
    ...actual,
    credentialUpdateCall: vi.fn(),
    credentialDeleteCall: vi.fn(),
  };
});

// Stub the edit modal so the panel's submit handler can be driven directly: the
// button fires onSubmit with form-shaped values, and it only renders when open.
vi.mock("./CredentialModal", () => ({
  default: function CredentialModalMock({
    mode,
    open,
    onSubmit,
  }: {
    mode: "add" | "edit";
    open: boolean;
    onSubmit: (values: Record<string, unknown>) => void;
  }) {
    if (!open) {
      return null;
    }
    const values = {
      credential_name: "openai-key",
      custom_llm_provider: "openai",
      api_key: "sk-1****2345",
      api_base: "https://proxy.e2e.example.com/v1",
    };
    return (
      <button data-testid={`credential-modal-${mode}-submit`} onClick={() => onSubmit(values)}>
        submit {mode}
      </button>
    );
  },
}));

vi.mock("./add_credential_wizard/AddCredentialWizard", () => ({
  default: function AddCredentialWizardMock({ onClose }: { onClose: () => void }) {
    return (
      <button data-testid="add-credential-wizard-close" onClick={onClose}>
        close wizard
      </button>
    );
  },
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
      <CredentialsPanel />
    </QueryClientProvider>,
  );

describe("CredentialsPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

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

  it("opens the Add Credential wizard, not the credential form, when the add button is clicked", async () => {
    const user = userEvent.setup();
    mockUseAuthorized.mockReturnValue({ accessToken: "test-token", userRole: "Admin" });
    mockUseCredentials.mockReturnValue({ data: { credentials: [] }, isLoading: false, refetch: vi.fn() });

    renderPanel();

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /add credential/i }));
    const dialog = await screen.findByRole("dialog", { name: "Add Credential" });
    expect(within(dialog).getByTestId("add-credential-wizard-close")).toBeInTheDocument();
    expect(screen.queryByTestId("credential-modal-add-submit")).not.toBeInTheDocument();
  });

  it("closes the wizard dialog when the wizard finishes", async () => {
    const user = userEvent.setup();
    mockUseAuthorized.mockReturnValue({ accessToken: "test-token", userRole: "Admin" });
    mockUseCredentials.mockReturnValue({ data: { credentials: [] }, isLoading: false, refetch: vi.fn() });

    renderPanel();

    await user.click(screen.getByRole("button", { name: /add credential/i }));
    await user.click(await screen.findByTestId("add-credential-wizard-close"));

    await waitFor(() => {
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    });
  });

  it("drops the masked api key from the update payload while keeping the edited api base", async () => {
    const user = userEvent.setup();
    mockUseAuthorized.mockReturnValue({ accessToken: "test-token", userRole: "Admin" });
    mockUseCredentials.mockReturnValue({ data: { credentials }, isLoading: false, refetch: vi.fn() });
    vi.mocked(credentialUpdateCall).mockResolvedValueOnce(undefined as never);

    renderPanel();

    await user.click(screen.getByTestId("credential-actions-openai-key"));
    await user.click(await screen.findByTestId("credential-action-edit"));
    await user.click(screen.getByTestId("credential-modal-edit-submit"));

    await waitFor(() => {
      expect(credentialUpdateCall).toHaveBeenCalled();
    });
    const [, updatedName, payload] = vi.mocked(credentialUpdateCall).mock.calls[0];
    expect(updatedName).toBe("openai-key");
    expect(payload.credential_values).toEqual({ api_base: "https://proxy.e2e.example.com/v1" });
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
