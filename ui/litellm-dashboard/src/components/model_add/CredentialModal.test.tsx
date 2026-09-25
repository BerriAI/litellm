import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent, { PointerEventsCheckLevel } from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { Providers } from "../provider_info_helpers";
import { CredentialItem } from "../networking";
import CredentialModal from "./CredentialModal";

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

  describe("credential alias", () => {
    const fillRequiredAddFields = async (user: ReturnType<typeof userEvent.setup>) => {
      fireEvent.change(screen.getByLabelText("Credential Name:"), { target: { value: "new-cred" } });
      const providerInput = await screen.findByPlaceholderText("Select a provider");
      await user.click(providerInput);
      await user.click(await screen.findByText("OpenAI"));
      fireEvent.change(await screen.findByLabelText("OpenAI API Key"), { target: { value: "sk-test" } });
    };

    it("submits the alias with the name in add mode", async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      const onSubmit = vi.fn();
      renderModal({ mode: "add", onSubmit });

      await fillRequiredAddFields(user);
      fireEvent.change(screen.getByLabelText("Alias:"), { target: { value: "Prod" } });
      fireEvent.click(screen.getByRole("button", { name: "Add Credential" }));

      await waitFor(() => expect(onSubmit).toHaveBeenCalled());
      expect(onSubmit.mock.calls[0][0]).toMatchObject({ credential_name: "new-cred", credential_alias: "Prod" });
    });

    it("submits no credential_alias key when the alias is left blank in add mode", async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      const onSubmit = vi.fn();
      renderModal({ mode: "add", onSubmit });

      await fillRequiredAddFields(user);
      fireEvent.click(screen.getByRole("button", { name: "Add Credential" }));

      await waitFor(() => expect(onSubmit).toHaveBeenCalled());
      expect(onSubmit.mock.calls[0][0]).not.toHaveProperty("credential_alias");
    });

    it("prefills the alias and submits an edited alias", async () => {
      const onSubmit = vi.fn();
      renderModal({
        mode: "edit",
        onSubmit,
        existingCredential: { ...mockCredential, credential_alias: "Prod" },
      });

      const aliasInput = screen.getByLabelText("Alias:") as HTMLInputElement;
      expect(aliasInput.value).toBe("Prod");

      fireEvent.change(aliasInput, { target: { value: "Staging" } });
      fireEvent.click(screen.getByRole("button", { name: "Update Credential" }));

      await waitFor(() => expect(onSubmit).toHaveBeenCalled());
      expect(onSubmit.mock.calls[0][0]).toMatchObject({
        credential_name: "test-credential",
        credential_alias: "Staging",
      });
    });

    it("submits credential_alias: null when the alias is cleared in edit mode", async () => {
      const onSubmit = vi.fn();
      renderModal({
        mode: "edit",
        onSubmit,
        existingCredential: { ...mockCredential, credential_alias: "Prod" },
      });

      fireEvent.change(screen.getByLabelText("Alias:"), { target: { value: "" } });
      fireEvent.click(screen.getByRole("button", { name: "Update Credential" }));

      await waitFor(() => expect(onSubmit).toHaveBeenCalled());
      expect(onSubmit.mock.calls[0][0].credential_alias).toBeNull();
    });
  });
});
