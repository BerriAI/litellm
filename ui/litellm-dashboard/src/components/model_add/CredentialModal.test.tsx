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

  describe("display name", () => {
    const fillRequiredAddFields = async (user: ReturnType<typeof userEvent.setup>) => {
      fireEvent.change(screen.getByLabelText("Credential Name:"), { target: { value: "new-cred" } });
      const providerInput = await screen.findByPlaceholderText("Select a provider");
      await user.click(providerInput);
      await user.click(await screen.findByText("OpenAI"));
      fireEvent.change(await screen.findByLabelText("OpenAI API Key"), { target: { value: "sk-test" } });
    };

    it("submits the display name with the credential name in add mode", async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      const onSubmit = vi.fn();
      renderModal({ mode: "add", onSubmit });

      await fillRequiredAddFields(user);
      fireEvent.change(screen.getByLabelText("Display Name:"), { target: { value: "Prod" } });
      fireEvent.click(screen.getByRole("button", { name: "Add Credential" }));

      await waitFor(() => expect(onSubmit).toHaveBeenCalled());
      expect(onSubmit.mock.calls[0][0]).toMatchObject({ credential_name: "new-cred", display_name: "Prod" });
    });

    it("submits no display_name key when the display name is left blank in add mode", async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      const onSubmit = vi.fn();
      renderModal({ mode: "add", onSubmit });

      await fillRequiredAddFields(user);
      fireEvent.click(screen.getByRole("button", { name: "Add Credential" }));

      await waitFor(() => expect(onSubmit).toHaveBeenCalled());
      expect(onSubmit.mock.calls[0][0]).not.toHaveProperty("display_name");
    });

    it("prefills the display name and submits an edited one", async () => {
      const onSubmit = vi.fn();
      renderModal({
        mode: "edit",
        onSubmit,
        existingCredential: { ...mockCredential, display_name: "Prod" },
      });

      const displayNameInput = screen.getByLabelText("Display Name:") as HTMLInputElement;
      expect(displayNameInput.value).toBe("Prod");

      fireEvent.change(displayNameInput, { target: { value: "Staging" } });
      fireEvent.click(screen.getByRole("button", { name: "Update Credential" }));

      await waitFor(() => expect(onSubmit).toHaveBeenCalled());
      expect(onSubmit.mock.calls[0][0]).toMatchObject({
        credential_name: "test-credential",
        display_name: "Staging",
      });
    });

    it("submits display_name: null when the display name is cleared in edit mode", async () => {
      const onSubmit = vi.fn();
      renderModal({
        mode: "edit",
        onSubmit,
        existingCredential: { ...mockCredential, display_name: "Prod" },
      });

      fireEvent.change(screen.getByLabelText("Display Name:"), { target: { value: "" } });
      fireEvent.click(screen.getByRole("button", { name: "Update Credential" }));

      await waitFor(() => expect(onSubmit).toHaveBeenCalled());
      expect(onSubmit.mock.calls[0][0].display_name).toBeNull();
    });

    it("trims the display name before submitting in add mode", async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      const onSubmit = vi.fn();
      renderModal({ mode: "add", onSubmit });

      await fillRequiredAddFields(user);
      fireEvent.change(screen.getByLabelText("Display Name:"), { target: { value: "  Prod  " } });
      fireEvent.click(screen.getByRole("button", { name: "Add Credential" }));

      await waitFor(() => expect(onSubmit).toHaveBeenCalled());
      expect(onSubmit.mock.calls[0][0]).toMatchObject({ credential_name: "new-cred", display_name: "Prod" });
    });

    it("submits display_name: null when the display name is whitespace-only in edit mode", async () => {
      const onSubmit = vi.fn();
      renderModal({
        mode: "edit",
        onSubmit,
        existingCredential: { ...mockCredential, display_name: "Prod OpenAI" },
      });

      fireEvent.change(screen.getByLabelText("Display Name:"), { target: { value: "   " } });
      fireEvent.click(screen.getByRole("button", { name: "Update Credential" }));

      await waitFor(() => expect(onSubmit).toHaveBeenCalled());
      expect(onSubmit.mock.calls[0][0].display_name).toBeNull();
    });

    it("leaves display_name out of the edit when it was not changed", async () => {
      const onSubmit = vi.fn();
      renderModal({
        mode: "edit",
        onSubmit,
        existingCredential: { ...mockCredential, display_name: "Prod" },
      });

      fireEvent.change(screen.getByLabelText("Display Name:"), { target: { value: " Prod " } });
      fireEvent.click(screen.getByRole("button", { name: "Update Credential" }));

      await waitFor(() => expect(onSubmit).toHaveBeenCalled());
      expect(onSubmit.mock.calls[0][0]).not.toHaveProperty("display_name");
    });

    it("accepts a 255-character display name and blocks a 256-character one", async () => {
      const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
      const onSubmit = vi.fn();
      renderModal({ mode: "add", onSubmit });

      await fillRequiredAddFields(user);
      fireEvent.change(screen.getByLabelText("Display Name:"), { target: { value: "x".repeat(256) } });
      fireEvent.click(screen.getByRole("button", { name: "Add Credential" }));
      expect(await screen.findByText("Display name must be at most 255 characters")).toBeInTheDocument();
      expect(onSubmit).not.toHaveBeenCalled();

      fireEvent.change(screen.getByLabelText("Display Name:"), { target: { value: "x".repeat(255) } });
      fireEvent.click(screen.getByRole("button", { name: "Add Credential" }));
      await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1));
      expect(onSubmit.mock.calls[0][0].display_name).toBe("x".repeat(255));
    });

    it("keeps the credential name read-only while the display name stays editable in edit mode", () => {
      renderModal({ mode: "edit", existingCredential: { ...mockCredential, display_name: "Prod" } });

      expect(screen.getByLabelText("Credential Name:")).toBeDisabled();
      expect(screen.getByLabelText("Display Name:")).toBeEnabled();
    });
  });
});
