import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { Providers } from "../provider_info_helpers";
import { CredentialItem } from "../networking";
import CredentialModal from "./CredentialModal";
import { chooseSelectOption } from "../../../tests/test-utils";

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
            { key: "anthropic_issuer_url", label: "Issuer URL", field_type: "text", required: true },
            { key: "anthropic_issuer_subject", label: "Subject", field_type: "text", required: true },
            { key: "anthropic_issuer_signing_key_ref", label: "Signing Key Reference", field_type: "text" },
            {
              key: "anthropic_identity_token",
              label: "Identity Token Reference",
              field_type: "text",
              required: true,
            },
          ],
          variants: [
            { id: "api_key", label: "API Key", field_keys: ["api_key"], fixed_values: {}, credential_only: false },
            {
              id: "wif_token",
              label: "Workload Identity Federation",
              field_keys: ["anthropic_federation_rule_id", "anthropic_organization_id", "anthropic_identity_token"],
              fixed_values: {},
              credential_only: true,
            },
            {
              id: "wif_internal_issuer",
              label: "Workload Identity Federation (LiteLLM-signed)",
              field_keys: [
                "anthropic_issuer_url",
                "anthropic_issuer_subject",
                "anthropic_issuer_signing_key_ref",
                "anthropic_organization_id",
                "anthropic_federation_rule_id",
              ],
              fixed_values: { anthropic_identity_source: "internal_issuer" },
              credential_only: true,
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
      <CredentialModal
        open={true}
        mode="add"
        onCancel={vi.fn()}
        onSubmit={vi.fn()}
        testConnection={vi.fn().mockResolvedValue({ models: [] })}
        loadJwks={vi.fn().mockResolvedValue({ keys: [] })}
        {...props}
      />
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

  describe("footer", () => {
    it("puts Cancel on the left and Test Connection beside the submit button on the right", () => {
      renderModal({ mode: "add" });

      const cancel = screen.getByRole("button", { name: "Cancel" });
      const testConnection = screen.getByRole("button", { name: "Test Connection" });
      const submit = screen.getByRole("button", { name: "Add Credential" });
      expect(cancel.compareDocumentPosition(testConnection) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
      expect(testConnection.compareDocumentPosition(submit) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    });

    it("closes on Cancel without submitting the form", () => {
      const onCancel = vi.fn();
      const onSubmit = vi.fn();
      renderModal({ mode: "add", onCancel, onSubmit });

      fireEvent.click(screen.getByRole("button", { name: "Cancel" }));

      expect(onCancel).toHaveBeenCalledTimes(1);
      expect(onSubmit).not.toHaveBeenCalled();
    });
  });

  describe("test connection", () => {
    it("tests an entered API key inline before the credential is saved", async () => {
      const testConnection = vi.fn().mockResolvedValue({ models: ["gpt-5.5", "gpt-5.5-mini"] });
      renderModal({ mode: "add", testConnection });

      fireEvent.change(await screen.findByLabelText("OpenAI API Key"), { target: { value: "sk-test" } });
      const button = screen.getByRole("button", { name: "Test Connection" });
      await waitFor(() => expect(button).toBeEnabled());
      fireEvent.click(button);

      expect(await screen.findByRole("alert")).toHaveTextContent(
        "Connection succeeded. 2 models available: gpt-5.5, gpt-5.5-mini.",
      );
      expect(testConnection).toHaveBeenCalledWith({ custom_llm_provider: "openai", api_key: "sk-test" });
    });

    it("keeps workload identity values out of an inline test and points at saving first", async () => {
      const testConnection = vi.fn();
      const user = userEvent.setup();
      renderModal({ mode: "add", testConnection });

      const providerInput = screen.getByRole("combobox", { name: "Provider:" });
      await user.click(providerInput);
      fireEvent.change(providerInput, { target: { value: "Anthropic" } });
      await user.click(await screen.findByRole("option", { name: (_, option) => option.textContent === "Anthropic" }));
      await chooseSelectOption(
        user,
        await screen.findByRole("combobox", { name: "Authentication method" }),
        "Workload Identity Federation (LiteLLM-signed)",
      );
      fireEvent.change(await screen.findByLabelText("Issuer URL"), {
        target: { value: "https://litellm.example.com" },
      });

      expect(await screen.findByText(/^Add the credential first, then test it from Edit/)).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Test Connection" })).toBeDisabled();
      expect(testConnection).not.toHaveBeenCalled();
    });

    it("tests a saved credential by name and surfaces the proxy error", async () => {
      const testConnection = vi.fn().mockRejectedValue(new Error("Model discovery failed: invalid x-api-key"));
      renderModal({ mode: "edit", existingCredential: mockCredential, testConnection });

      await screen.findByLabelText("OpenAI API Key");
      fireEvent.click(screen.getByRole("button", { name: "Test Connection" }));

      expect(await screen.findByRole("alert")).toHaveTextContent("Model discovery failed: invalid x-api-key");
      expect(testConnection).toHaveBeenCalledWith({
        custom_llm_provider: "openai",
        litellm_credential_name: "test-credential",
      });
    });

    it("waits for unsaved edits to be saved before testing a saved credential", async () => {
      const testConnection = vi.fn();
      renderModal({ mode: "edit", existingCredential: mockCredential, testConnection });

      fireEvent.change(await screen.findByLabelText("OpenAI API Key"), { target: { value: "sk-changed" } });

      expect(
        await screen.findByText("Update the credential first. Test Connection checks the saved values."),
      ).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Test Connection" })).toBeDisabled();
      expect(testConnection).not.toHaveBeenCalled();
    });
  });

  describe("LiteLLM-signed JWKS", () => {
    const signedCredential: CredentialItem = {
      credential_name: "anthropic-signed",
      credential_values: {
        anthropic_identity_source: "internal_issuer",
        anthropic_issuer_url: "https://litellm.example.com",
        anthropic_issuer_subject: "litellm-proxy",
        anthropic_issuer_signing_key_ref: "os.environ/ISSUER_SIGNING_KEY_PEM",
        anthropic_organization_id: "",
        anthropic_federation_rule_id: "",
      },
      credential_info: { custom_llm_provider: Providers.Anthropic },
    };

    it("shows the saved credential's public JWKS with a copy button", async () => {
      const loadJwks = vi.fn().mockResolvedValue({ keys: [{ kty: "RSA", kid: "k-2026", n: "abc", e: "AQAB" }] });
      renderModal({ mode: "edit", existingCredential: signedCredential, loadJwks });

      expect(await screen.findByText(/"kid": "k-2026"/)).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Copy JWKS" })).toBeInTheDocument();
      expect(loadJwks).toHaveBeenCalledWith("anthropic-signed");
    });

    it("does not request a JWKS for a credential that is not LiteLLM-signed", async () => {
      const loadJwks = vi.fn();
      renderModal({ mode: "edit", existingCredential: mockCredential, loadJwks });

      await screen.findByLabelText("OpenAI API Key");

      expect(screen.queryByText("Public JWKS")).not.toBeInTheDocument();
      expect(loadJwks).not.toHaveBeenCalled();
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

  const presetWifProps = { mode: "add" as const, initialProvider: Providers.Anthropic, initialVariantId: "wif_token" };

  describe("after submit", () => {
    it("keeps the typed values when saving fails", async () => {
      const onSubmit = vi.fn().mockResolvedValue(false);
      renderModal({ ...presetWifProps, onSubmit });
      const user = userEvent.setup();

      await user.type(screen.getByLabelText("Credential Name:"), "anthropic-wif");
      await user.type(await screen.findByLabelText("Federation Rule ID"), "rule-1");
      await user.type(screen.getByLabelText("Organization ID"), "org-1");
      await user.type(screen.getByLabelText("Identity Token Reference"), "oidc/env/TOKEN");
      await user.click(screen.getByRole("button", { name: "Add Credential" }));

      await waitFor(() => expect(onSubmit).toHaveBeenCalled());
      expect(screen.getByLabelText("Credential Name:")).toHaveValue("anthropic-wif");
      expect(screen.getByLabelText("Federation Rule ID")).toHaveValue("rule-1");
      expect(screen.getByLabelText("Identity Token Reference")).toHaveValue("oidc/env/TOKEN");
    });

    it("clears the form once saving succeeds", async () => {
      const onSubmit = vi.fn().mockResolvedValue(true);
      renderModal({ ...presetWifProps, onSubmit });
      const user = userEvent.setup();

      await user.type(screen.getByLabelText("Credential Name:"), "anthropic-wif");
      await user.type(await screen.findByLabelText("Federation Rule ID"), "rule-1");
      await user.type(screen.getByLabelText("Organization ID"), "org-1");
      await user.type(screen.getByLabelText("Identity Token Reference"), "oidc/env/TOKEN");
      await user.click(screen.getByRole("button", { name: "Add Credential" }));

      await waitFor(() => expect(onSubmit).toHaveBeenCalled());
      await waitFor(() => expect(screen.getByLabelText("Credential Name:")).toHaveValue(""));
    });
  });

  describe("preset from the Add Model form", () => {
    it("opens on the handed-over provider and variant and submits that provider", async () => {
      const onSubmit = vi.fn();
      renderModal({ ...presetWifProps, onSubmit });
      const user = userEvent.setup();

      expect(await screen.findByLabelText("Federation Rule ID")).toBeInTheDocument();
      expect(screen.queryByLabelText("Anthropic API Key")).not.toBeInTheDocument();
      expect(screen.queryByLabelText("OpenAI API Key")).not.toBeInTheDocument();

      await user.type(screen.getByLabelText("Credential Name:"), "anthropic-wif");
      await user.type(screen.getByLabelText("Federation Rule ID"), "rule-1");
      await user.type(screen.getByLabelText("Organization ID"), "org-1");
      await user.type(screen.getByLabelText("Identity Token Reference"), "oidc/env/TOKEN");
      await user.click(screen.getByRole("button", { name: "Add Credential" }));

      await waitFor(() => expect(onSubmit).toHaveBeenCalled());
      const expectedCredential = {
        credential_name: "anthropic-wif",
        custom_llm_provider: "Anthropic",
        anthropic_federation_rule_id: "rule-1",
        anthropic_organization_id: "org-1",
        anthropic_identity_token: "oidc/env/TOKEN",
      };
      expect(onSubmit.mock.calls[0][0]).toEqual(expectedCredential);
    });
  });
});
