import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { Providers } from "../provider_info_helpers";
import { CredentialItem, type ChatGPTLoginStartRequest } from "../networking";
import CredentialModal from "./CredentialModal";

const signInModal = vi.hoisted(() => vi.fn());

vi.mock("./ChatGPTSignInModal", () => ({
  default: (props: { credentialName: string; request: ChatGPTLoginStartRequest; onConnected: () => void }) => {
    signInModal(props);
    return (
      <div data-testid="chatgpt-sign-in">
        <button onClick={props.onConnected}>Finish ChatGPT sign-in</button>
      </div>
    );
  },
}));

vi.mock("../common_components/team_dropdown", () => ({
  default: ({ id, value, onChange }: { id?: string; value?: string | null; onChange?: (v: string | null) => void }) => (
    <select id={id} value={value ?? ""} onChange={(event) => onChange?.(event.target.value || null)}>
      <option value="">No team</option>
      <option value="team-1">Team One</option>
    </select>
  ),
}));

vi.mock("../networking", async () => {
  const actual = await vi.importActual("../networking");
  return {
    ...actual,
    getProviderCreateMetadata: vi.fn().mockResolvedValue([
      {
        provider: "OpenAI",
        provider_display_name: Providers.OpenAI,
        litellm_provider: "openai",
        default_model_placeholder: "gpt-3.5-turbo",
        credential_fields: [
          {
            key: "api_key",
            label: "OpenAI API Key",
            field_type: "password",
            required: true,
          },
          {
            key: "api_base",
            label: "API Base",
            field_type: "text",
            placeholder: "https://api.openai.com/v1",
          },
        ],
      },
      {
        provider: "Anthropic",
        provider_display_name: Providers.Anthropic,
        litellm_provider: "anthropic",
        default_model_placeholder: "claude-3-opus-20240229",
        credential_fields: [
          {
            key: "api_key",
            label: "Anthropic API Key",
            field_type: "password",
            required: true,
          },
        ],
      },
    ]),
  };
});

const createQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
      },
    },
  });

const mockCredential: CredentialItem = {
  credential_name: "test-credential",
  credential_values: {
    api_key: "test-api-key",
    api_base: "https://api.test.com",
  },
  credential_info: {
    custom_llm_provider: Providers.OpenAI,
  },
};

const renderModal = (props: Partial<React.ComponentProps<typeof CredentialModal>> = {}) =>
  render(
    <QueryClientProvider client={createQueryClient()}>
      <CredentialModal open={true} mode="add" onCancel={vi.fn()} onSubmit={vi.fn()} {...props} />
    </QueryClientProvider>,
  );

describe("CredentialModal", () => {
  describe("add mode", () => {
    it("renders the add title and an editable credential name", () => {
      renderModal({ mode: "add" });

      expect(screen.getByText("Add New Credential")).toBeInTheDocument();
      expect(screen.getByText("Add Credential")).toBeInTheDocument();
      const nameInput = screen.getByLabelText("Credential Name:") as HTMLInputElement;
      expect(nameInput.value).toBe("");
      expect(nameInput).toBeEnabled();
    });

    it("shows provider-specific fields for the selected provider", async () => {
      renderModal({ mode: "add" });

      await waitFor(() => {
        expect(screen.getByLabelText("OpenAI API Key")).toBeInTheDocument();
        expect(screen.getByPlaceholderText("https://api.openai.com/v1")).toBeInTheDocument();
      });
    });
  });

  describe("edit mode", () => {
    it("renders the edit title and update button", () => {
      renderModal({ mode: "edit", existingCredential: mockCredential });

      expect(screen.getByText("Edit Credential")).toBeInTheDocument();
      expect(screen.getByText("Update Credential")).toBeInTheDocument();
    });

    it("prefills the credential name and disables it", async () => {
      renderModal({ mode: "edit", existingCredential: mockCredential });

      await waitFor(() => {
        const nameInput = screen.getByLabelText("Credential Name:") as HTMLInputElement;
        expect(nameInput.value).toBe("test-credential");
        expect(nameInput).toBeDisabled();
      });
    });

    it("disables the name from the mode, not the credential's name value", () => {
      renderModal({
        mode: "edit",
        existingCredential: { ...mockCredential, credential_name: "" },
      });

      expect(screen.getByLabelText("Credential Name:")).toBeDisabled();
    });
  });

  describe("ChatGPT Subscription", () => {
    beforeEach(() => {
      signInModal.mockClear();
    });

    const lastSignIn = () =>
      signInModal.mock.lastCall?.[0] as { credentialName: string; request: ChatGPTLoginStartRequest } | undefined;

    const chatGPTCredential: CredentialItem = {
      credential_name: "key-chatgpt",
      display_name: "Key ChatGPT",
      credential_values: { chatgpt_access_token: "eyJh****", chatgpt_account_id: "acct-1" },
      credential_info: { custom_llm_provider: "chatgpt", chatgpt_owner_key_hash: "hash-stored" },
    };

    it("offers sign-in and an owner picker in place of the add button", () => {
      renderModal({ mode: "add", initialProvider: "CHATGPT" });

      expect(screen.getByRole("button", { name: "Sign in with ChatGPT" })).toBeInTheDocument();
      expect(screen.queryByRole("button", { name: "Add Credential" })).not.toBeInTheDocument();
      expect(screen.getByRole("radio", { name: "Team" })).toBeChecked();
      expect(screen.getByLabelText("Team:")).toBeInTheDocument();
    });

    it("signs in for the entered name and team without calling onSubmit, then reports the sign-in", async () => {
      const user = userEvent.setup();
      const onSubmit = vi.fn();
      const onSignedIn = vi.fn();
      renderModal({ initialProvider: "CHATGPT", onSubmit, onSignedIn });

      fireEvent.change(screen.getByLabelText("Credential Name:"), { target: { value: "team-chatgpt" } });
      fireEvent.change(screen.getByLabelText("Team:"), { target: { value: "team-1" } });
      await user.click(screen.getByRole("button", { name: "Sign in with ChatGPT" }));

      expect(await screen.findByTestId("chatgpt-sign-in")).toBeInTheDocument();
      expect(lastSignIn()).toMatchObject({
        credentialName: "team-chatgpt",
        request: { ownerTeamId: "team-1", ownerKeyHash: null, displayName: null },
      });

      await user.click(screen.getByRole("button", { name: "Finish ChatGPT sign-in" }));
      expect(onSignedIn).toHaveBeenCalledTimes(1);
      expect(screen.queryByTestId("chatgpt-sign-in")).not.toBeInTheDocument();
      expect(onSubmit).not.toHaveBeenCalled();
    });

    it("requires an owner and sends exactly one of the team or the virtual key", async () => {
      const user = userEvent.setup();
      renderModal({ mode: "add", initialProvider: "CHATGPT" });
      fireEvent.change(screen.getByLabelText("Credential Name:"), { target: { value: "key-chatgpt" } });

      await user.click(screen.getByRole("button", { name: "Sign in with ChatGPT" }));
      expect(await screen.findByText("Pick the team that owns this login")).toBeInTheDocument();
      expect(signInModal).not.toHaveBeenCalled();

      fireEvent.change(screen.getByLabelText("Team:"), { target: { value: "team-1" } });
      await user.click(screen.getByRole("radio", { name: "Virtual key" }));
      await user.click(screen.getByRole("button", { name: "Sign in with ChatGPT" }));
      expect(await screen.findByText("Enter the ID of the key that owns this login")).toBeInTheDocument();
      expect(signInModal).not.toHaveBeenCalled();

      fireEvent.change(screen.getByLabelText("Key ID:"), { target: { value: " hash-1 " } });
      await user.click(screen.getByRole("button", { name: "Sign in with ChatGPT" }));

      expect(await screen.findByTestId("chatgpt-sign-in")).toBeInTheDocument();
      expect(lastSignIn()?.request).toEqual({ ownerTeamId: null, ownerKeyHash: "hash-1", displayName: null });
    });

    it("recognizes a stored chatgpt credential and signs in again with its saved owner and name", async () => {
      const user = userEvent.setup();
      const onSubmit = vi.fn();
      renderModal({ mode: "edit", existingCredential: chatGPTCredential, onSubmit });

      expect(screen.getByRole("combobox", { name: /Provider/ })).toHaveValue(Providers.CHATGPT);
      expect(screen.getByRole("radio", { name: "Virtual key" })).toBeChecked();
      expect(screen.getByLabelText("Key ID:")).toHaveValue("hash-stored");

      await user.click(screen.getByRole("button", { name: "Sign in again" }));

      expect(await screen.findByTestId("chatgpt-sign-in")).toBeInTheDocument();
      expect(lastSignIn()).toMatchObject({
        credentialName: "key-chatgpt",
        request: { ownerTeamId: null, ownerKeyHash: "hash-stored", displayName: "Key ChatGPT" },
      });
      expect(onSubmit).not.toHaveBeenCalled();
    });
  });
});
