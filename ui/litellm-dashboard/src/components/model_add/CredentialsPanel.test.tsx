import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { CredentialItem, credentialCreateCall, credentialUpdateCall } from "@/components/networking";
import { toast } from "@/lib/toast";

import CredentialsPanel from "./CredentialsPanel";
import type { CredentialSubmission } from "./credential_form_helpers";

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
    credentialCreateCall: vi.fn(),
    credentialUpdateCall: vi.fn(),
    credentialDeleteCall: vi.fn(),
  };
});

// Stub the modal so the panel's submit handlers can be driven directly: the
// button fires onSubmit with form-shaped values, and it only renders when open.
vi.mock("./CredentialModal", () => ({
  default: function CredentialModalMock({
    mode,
    open,
    onSubmit,
  }: {
    mode: "add" | "edit";
    open: boolean;
    onSubmit: (submission: CredentialSubmission, valuesToDelete?: readonly string[]) => void;
  }) {
    if (!open) {
      return null;
    }
    const values: CredentialSubmission =
      mode === "edit"
        ? {
            credential_name: "openai-key",
            custom_llm_provider: "openai",
            credential_values: {
              api_key: "sk-1****2345",
              api_base: "https://proxy.e2e.example.com/v1",
              credential_name: "other-credential",
            },
          }
        : { credential_name: "new-cred", custom_llm_provider: "openai", credential_values: {} };
    return (
      <>
        <button data-testid={`credential-modal-${mode}-submit`} onClick={() => onSubmit(values)}>
          submit {mode}
        </button>
        <button
          data-testid={`credential-modal-${mode}-switch-to-federation`}
          onClick={() =>
            onSubmit(
              {
                ...values,
                credential_values: {
                  ...values.credential_values,
                  api_key: undefined,
                  anthropic_federation_rule_id: "fdrl_new",
                },
              },
              ["api_key"],
            )
          }
        >
          switch {mode} to federation
        </button>
      </>
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

  it("opens the add modal when the add button is clicked", async () => {
    const user = userEvent.setup();
    mockUseAuthorized.mockReturnValue({ accessToken: "test-token", userRole: "Admin" });
    mockUseCredentials.mockReturnValue({ data: { credentials: [] }, isLoading: false, refetch: vi.fn() });

    renderPanel();

    expect(screen.queryByTestId("credential-modal-add-submit")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /add credential/i }));
    expect(screen.getByTestId("credential-modal-add-submit")).toBeInTheDocument();
  });

  it("closes the add modal and refetches after a successful add", async () => {
    const user = userEvent.setup();
    const refetch = vi.fn();
    mockUseAuthorized.mockReturnValue({ accessToken: "test-token", userRole: "Admin" });
    mockUseCredentials.mockReturnValue({ data: { credentials: [] }, isLoading: false, refetch });
    vi.mocked(credentialCreateCall).mockResolvedValueOnce(undefined as never);

    renderPanel();

    await user.click(screen.getByRole("button", { name: /add credential/i }));
    await user.click(screen.getByTestId("credential-modal-add-submit"));

    await waitFor(() => {
      expect(toast.success).toHaveBeenCalledWith("Credential added successfully");
    });
    expect(refetch).toHaveBeenCalled();
    expect(screen.queryByTestId("credential-modal-add-submit")).not.toBeInTheDocument();
  });

  it("surfaces an error and keeps the add modal open when the create call fails", async () => {
    const user = userEvent.setup();
    mockUseAuthorized.mockReturnValue({ accessToken: "test-token", userRole: "Admin" });
    mockUseCredentials.mockReturnValue({ data: { credentials: [] }, isLoading: false, refetch: vi.fn() });
    vi.mocked(credentialCreateCall).mockRejectedValueOnce(
      new Error("Credential 'new-cred' is defined in config. Edit config.yaml to change it."),
    );

    renderPanel();

    await user.click(screen.getByRole("button", { name: /add credential/i }));
    await user.click(screen.getByTestId("credential-modal-add-submit"));

    await waitFor(() => {
      expect(toast.error).toHaveBeenCalledWith(
        "Credential 'new-cred' is defined in config. Edit config.yaml to change it.",
      );
    });
    // The modal stays open so the user can retry, and no success toast fired.
    expect(screen.getByTestId("credential-modal-add-submit")).toBeInTheDocument();
    expect(toast.success).not.toHaveBeenCalled();
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
    expect(payload.credential_name).toBe("openai-key");
    expect(payload.credential_values).toEqual({
      api_base: "https://proxy.e2e.example.com/v1",
      credential_name: "other-credential",
    });
  });

  it("sends the values to delete with the update and omits the field when there are none", async () => {
    const user = userEvent.setup();
    mockUseAuthorized.mockReturnValue({ accessToken: "test-token", userRole: "Admin" });
    mockUseCredentials.mockReturnValue({ data: { credentials }, isLoading: false, refetch: vi.fn() });
    vi.mocked(credentialUpdateCall).mockResolvedValue(undefined as never);

    renderPanel();

    await user.click(screen.getByTestId("credential-actions-openai-key"));
    await user.click(await screen.findByTestId("credential-action-edit"));
    await user.click(screen.getByTestId("credential-modal-edit-switch-to-federation"));
    await waitFor(() => {
      expect(credentialUpdateCall).toHaveBeenCalledTimes(1);
    });
    const [, , federatedPayload] = vi.mocked(credentialUpdateCall).mock.calls[0];
    expect(federatedPayload.credential_values).toEqual({
      api_base: "https://proxy.e2e.example.com/v1",
      credential_name: "other-credential",
      anthropic_federation_rule_id: "fdrl_new",
    });
    expect(federatedPayload.credential_values_to_delete).toEqual(["api_key"]);

    await user.click(screen.getByTestId("credential-actions-openai-key"));
    await user.click(await screen.findByTestId("credential-action-edit"));
    await user.click(screen.getByTestId("credential-modal-edit-submit"));
    await waitFor(() => {
      expect(credentialUpdateCall).toHaveBeenCalledTimes(2);
    });
    const [, , plainPayload] = vi.mocked(credentialUpdateCall).mock.calls[1];
    expect(plainPayload).not.toHaveProperty("credential_values_to_delete");
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
