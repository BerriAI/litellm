import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
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
        credential_variants: {
          selector_label: "Authentication method",
          default_variant: "api_key",
          field_definitions: [
            { key: "api_key", label: "Anthropic API Key", field_type: "password" },
            { key: "anthropic_federation_rule_id", label: "Federation Rule ID", field_type: "text", required: true },
            { key: "anthropic_organization_id", label: "Organization ID", field_type: "text", required: true },
            {
              key: "anthropic_identity_token",
              label: "Identity Token Reference",
              field_type: "text",
              required: true,
            },
          ],
          variants: [
            { id: "api_key", label: "API Key", field_keys: ["api_key"], fixed_values: {} },
            {
              id: "wif_token",
              label: "Workload Identity Federation",
              field_keys: ["anthropic_federation_rule_id", "anthropic_organization_id", "anthropic_identity_token"],
              fixed_values: {},
            },
          ],
        },
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

  describe("credential_values_to_delete", () => {
    const anthropicWifCredential: CredentialItem = {
      credential_name: "anthropic-wif-cred",
      credential_values: {
        anthropic_federation_rule_id: "rule-1",
        anthropic_organization_id: "org-1",
        anthropic_identity_token: "oidc/env/TOKEN",
      },
      credential_info: {
        custom_llm_provider: Providers.Anthropic,
      },
    };

    it("flags the previous variant's fields for deletion when the operator switches variants", async () => {
      const onSubmit = vi.fn();
      const user = userEvent.setup();
      renderModal({ mode: "edit", existingCredential: anthropicWifCredential, onSubmit });

      await screen.findByLabelText("Federation Rule ID");
      await user.click(await screen.findByRole("combobox", { name: "Authentication method" }));
      await user.click(await screen.findByRole("option", { name: "API Key" }));
      await screen.findByLabelText("Anthropic API Key");

      await user.click(screen.getByRole("button", { name: "Update Credential" }));

      await waitFor(() => expect(onSubmit).toHaveBeenCalled());
      const [, deletedKeys] = onSubmit.mock.calls[0];
      expect([...deletedKeys].sort()).toEqual([
        "anthropic_federation_rule_id",
        "anthropic_identity_token",
        "anthropic_organization_id",
      ]);
    });

    it("flags nothing for deletion when the variant and values are untouched", async () => {
      const onSubmit = vi.fn();
      renderModal({ mode: "edit", existingCredential: anthropicWifCredential, onSubmit });

      await screen.findByLabelText("Federation Rule ID");
      fireEvent.click(screen.getByRole("button", { name: "Update Credential" }));

      await waitFor(() => expect(onSubmit).toHaveBeenCalled());
      const [, deletedKeys] = onSubmit.mock.calls[0];
      expect(deletedKeys).toEqual([]);
    });
  });
});
