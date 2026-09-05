import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeAll, describe, expect, it, vi } from "vitest";
import { useFormContext } from "react-hook-form";
import { Providers } from "../provider_info_helpers";
import type { MountedFormValues } from "../common_components/MountedFormField";
import { MountedFormHost } from "../../../tests/mounted-form-host";
import ProviderSpecificFields from "./provider_specific_fields";

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
            key: "api_base",
            label: "API Base",
            field_type: "text",
            placeholder: "https://api.openai.com/v1",
            tooltip:
              "Common endpoints: https://api.openai.com/v1, https://eu.api.openai.com, https://us.api.openai.com",
            default_value: "https://api.openai.com/v1",
          },
          {
            key: "organization",
            label: "OpenAI Organization ID",
            placeholder: "[OPTIONAL] my-unique-org",
          },
          {
            key: "api_key",
            label: "OpenAI API Key",
            field_type: "password",
            required: true,
          },
        ],
      },
      {
        provider: "Vertex_AI",
        provider_display_name: Providers.Vertex_AI,
        litellm_provider: "vertex_ai",
        default_model_placeholder: "gemini-pro",
        credential_fields: [
          {
            key: "vertex_credentials",
            label: "Vertex Credentials",
            field_type: "upload",
          },
        ],
      },
      {
        provider: "Hosted_Vllm",
        provider_display_name: Providers.Hosted_Vllm,
        litellm_provider: "hosted_vllm",
        default_model_placeholder: "vllm/any-model",
        credential_fields: [
          {
            key: "api_base",
            label: "API Base",
            placeholder: "https://...",
          },
          {
            key: "api_key",
            label: "vLLM API Key",
            field_type: "password",
          },
        ],
      },
      {
        provider: "Azure",
        provider_display_name: Providers.Azure,
        litellm_provider: "azure",
        default_model_placeholder: "azure/my-deployment",
        credential_fields: [
          {
            key: "api_base",
            label: "API Base",
            placeholder: "https://...",
            required: true,
          },
          {
            key: "api_version",
            label: "API Version",
            placeholder: "2023-07-01-preview",
            tooltip:
              "By default litellm will use the latest version. If you want to use a different version, you can specify it here",
          },
          {
            key: "base_model",
            label: "Base Model",
            placeholder: "azure/gpt-3.5-turbo",
          },
          {
            key: "api_key",
            label: "Azure API Key",
            field_type: "password",
            placeholder: "Enter your Azure API Key",
          },
          {
            key: "azure_ad_token",
            label: "Azure AD Token",
            field_type: "password",
            placeholder: "Enter your Azure AD Token",
          },
        ],
      },
      {
        provider: "Anthropic",
        provider_display_name: Providers.Anthropic,
        litellm_provider: "anthropic",
        default_model_placeholder: "claude-3-opus",
        credential_fields: [
          { key: "api_base", label: "Upstream API Base", field_type: "text" },
          { key: "api_key", label: "API Key", field_type: "password" },
        ],
        credential_variants: {
          selector_label: "Authentication method",
          default_variant: "api_key",
          field_definitions: [
            { key: "api_base", label: "Upstream API Base", field_type: "text" },
            { key: "api_key", label: "API Key", field_type: "password" },
            { key: "anthropic_federation_rule_id", label: "Federation Rule ID", field_type: "text", required: true },
            { key: "anthropic_organization_id", label: "Organization ID", field_type: "text", required: true },
            {
              key: "anthropic_identity_token",
              label: "Identity Token Reference",
              field_type: "text",
              required: true,
            },
            {
              key: "anthropic_issuer_url",
              label: "Issuer URL",
              field_type: "text",
              required: true,
            },
            {
              key: "anthropic_issuer_signing_key_ref",
              label: "Signing Key Reference",
              field_type: "text",
              required: true,
              tooltip: "A secret REFERENCE, e.g. os.environ/VAR_NAME. Never the key itself.",
            },
          ],
          variants: [
            { id: "api_key", label: "API Key", field_keys: ["api_base", "api_key"], fixed_values: {} },
            {
              id: "wif_token",
              label: "Workload Identity Federation (external token)",
              field_keys: ["anthropic_federation_rule_id", "anthropic_organization_id", "anthropic_identity_token"],
              fixed_values: {},
            },
            {
              id: "wif_internal_issuer",
              label: "Workload Identity Federation (LiteLLM-signed)",
              field_keys: [
                "anthropic_federation_rule_id",
                "anthropic_organization_id",
                "anthropic_issuer_url",
                "anthropic_issuer_signing_key_ref",
              ],
              fixed_values: { anthropic_identity_source: "internal_issuer" },
            },
          ],
        },
      },
    ]),
  };
});

beforeAll(() => {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: (query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: () => {},
      removeListener: () => {},
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => false,
    }),
  });
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

const VertexCredentialsProbe = () => {
  const { watch } = useFormContext<MountedFormValues>();
  return <output data-testid="vertex-credentials">{String(watch("vertex_credentials") ?? "")}</output>;
};

describe("ProviderSpecificFields", () => {
  it("reads a picked service-account file into the vertex credentials field", async () => {
    const queryClient = createQueryClient();
    render(
      <QueryClientProvider client={queryClient}>
        <MountedFormHost>
          <ProviderSpecificFields selectedProvider={"Vertex_AI" as Providers} />
          <VertexCredentialsProbe />
        </MountedFormHost>
      </QueryClientProvider>,
    );

    const fileInput = await screen.findByLabelText("Vertex Credentials");
    const serviceAccount = '{"project_id":"example"}';
    fireEvent.change(fileInput, {
      target: { files: [new File([serviceAccount], "vertex.json", { type: "application/json" })] },
    });

    await waitFor(() => expect(screen.getByTestId("vertex-credentials")).toHaveTextContent(serviceAccount));
  });

  it("ignores a picked file that is not JSON", async () => {
    const queryClient = createQueryClient();
    render(
      <QueryClientProvider client={queryClient}>
        <MountedFormHost>
          <ProviderSpecificFields selectedProvider={"Vertex_AI" as Providers} />
          <VertexCredentialsProbe />
        </MountedFormHost>
      </QueryClientProvider>,
    );

    const fileInput = await screen.findByLabelText("Vertex Credentials");
    fireEvent.change(fileInput, {
      target: { files: [new File(["not json"], "vertex.txt", { type: "text/plain" })] },
    });

    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(screen.getByTestId("vertex-credentials")).toBeEmptyDOMElement();
  });

  it("should render", async () => {
    const queryClient = createQueryClient();
    render(
      <QueryClientProvider client={queryClient}>
        <MountedFormHost>
          <ProviderSpecificFields selectedProvider={Providers.OpenAI} />
        </MountedFormHost>
      </QueryClientProvider>,
    );

    await waitFor(() => {
      expect(screen.getByLabelText("OpenAI API Key")).toBeInTheDocument();
    });
  });

  it("should render the provider specific fields for OpenAI", async () => {
    const queryClient = createQueryClient();
    render(
      <QueryClientProvider client={queryClient}>
        <MountedFormHost>
          <ProviderSpecificFields selectedProvider={Providers.OpenAI} />
        </MountedFormHost>
      </QueryClientProvider>,
    );

    const apiKeyLabel = await screen.findByLabelText("OpenAI API Key");
    expect(apiKeyLabel).toBeInTheDocument();

    const apiBaseInput = screen.getByPlaceholderText("https://api.openai.com/v1");
    expect(apiBaseInput).toBeInTheDocument();
    expect(apiBaseInput).toHaveAttribute("type", "text");

    const orgInput = screen.getByPlaceholderText("[OPTIONAL] my-unique-org");
    expect(orgInput).toBeInTheDocument();
  });

  it("should let the user reveal a secret field", async () => {
    const queryClient = createQueryClient();
    render(
      <QueryClientProvider client={queryClient}>
        <MountedFormHost>
          <ProviderSpecificFields selectedProvider={Providers.OpenAI} />
        </MountedFormHost>
      </QueryClientProvider>,
    );

    const apiKeyInput = await screen.findByLabelText("OpenAI API Key");
    expect(apiKeyInput).toHaveAttribute("type", "password");

    fireEvent.click(screen.getByRole("button", { name: "Show password" }));
    expect(await screen.findByLabelText("OpenAI API Key")).toHaveAttribute("type", "text");

    fireEvent.click(screen.getByRole("button", { name: "Hide password" }));
    expect(await screen.findByLabelText("OpenAI API Key")).toHaveAttribute("type", "password");
  });

  it("should render the provider specific fields for vLLM", async () => {
    const queryClient = createQueryClient();
    render(
      <QueryClientProvider client={queryClient}>
        <MountedFormHost>
          <ProviderSpecificFields selectedProvider={"Hosted_Vllm" as Providers} />
        </MountedFormHost>
      </QueryClientProvider>,
    );

    const apiKeyLabel = await screen.findByLabelText("vLLM API Key");
    expect(apiKeyLabel).toBeInTheDocument();

    const apiBaseInput = screen.getByPlaceholderText("https://...");
    expect(apiBaseInput).toBeInTheDocument();
    expect(apiBaseInput).toHaveAttribute("type", "text");
  });

  it("should render the provider specific fields for Azure", async () => {
    const queryClient = createQueryClient();
    render(
      <QueryClientProvider client={queryClient}>
        <MountedFormHost>
          <ProviderSpecificFields selectedProvider={Providers.Azure} />
        </MountedFormHost>
      </QueryClientProvider>,
    );

    const apiKeyInput = await screen.findByLabelText("Azure API Key");
    expect(apiKeyInput).toBeInTheDocument();
    expect(apiKeyInput).toHaveAttribute("type", "password");
    expect(apiKeyInput).toHaveAttribute("placeholder", "Enter your Azure API Key");

    const azureAdTokenInput = screen.getByLabelText("Azure AD Token");
    expect(azureAdTokenInput).toBeInTheDocument();
    expect(azureAdTokenInput).toHaveAttribute("type", "password");
    expect(azureAdTokenInput).toHaveAttribute("placeholder", "Enter your Azure AD Token");

    const apiBaseInput = screen.getByPlaceholderText("https://...");
    expect(apiBaseInput).toBeInTheDocument();
    expect(apiBaseInput).toHaveAttribute("type", "text");

    const apiVersionInput = screen.getByPlaceholderText("2023-07-01-preview");
    expect(apiVersionInput).toBeInTheDocument();

    const baseModelInput = screen.getByPlaceholderText("azure/gpt-3.5-turbo");
    expect(baseModelInput).toBeInTheDocument();
  });

  it("sets Azure API version from the API base query parameter", async () => {
    const queryClient = createQueryClient();
    render(
      <QueryClientProvider client={queryClient}>
        <MountedFormHost>
          <ProviderSpecificFields selectedProvider={Providers.Azure} />
        </MountedFormHost>
      </QueryClientProvider>,
    );

    const apiBaseInput = await screen.findByPlaceholderText("https://...");
    const apiVersionInput = await screen.findByPlaceholderText("2023-07-01-preview");

    fireEvent.change(apiBaseInput, {
      target: {
        value:
          "https://test-resource.openai.azure.com/openai/deployments/gpt-4/chat/completions?api_version=2024-10-21",
      },
    });

    await waitFor(() => {
      expect(apiVersionInput).toHaveValue("2024-10-21");
    });
  });

  it("sets Azure API version from the hyphenated API base query parameter", async () => {
    const queryClient = createQueryClient();
    render(
      <QueryClientProvider client={queryClient}>
        <MountedFormHost>
          <ProviderSpecificFields selectedProvider={Providers.Azure} />
        </MountedFormHost>
      </QueryClientProvider>,
    );

    const apiBaseInput = await screen.findByPlaceholderText("https://...");
    const apiVersionInput = await screen.findByPlaceholderText("2023-07-01-preview");

    fireEvent.change(apiBaseInput, {
      target: {
        value:
          "https://test-resource.openai.azure.com/openai/deployments/gpt-4/chat/completions?api-version=2024-10-21",
      },
    });

    await waitFor(() => {
      expect(apiVersionInput).toHaveValue("2024-10-21");
    });
  });

  it("clears an inferred Azure API version when the API base has no version parameter", async () => {
    const queryClient = createQueryClient();
    render(
      <QueryClientProvider client={queryClient}>
        <MountedFormHost>
          <ProviderSpecificFields selectedProvider={Providers.Azure} />
        </MountedFormHost>
      </QueryClientProvider>,
    );

    const apiBaseInput = await screen.findByPlaceholderText("https://...");
    const apiVersionInput = await screen.findByPlaceholderText("2023-07-01-preview");

    fireEvent.change(apiBaseInput, {
      target: {
        value:
          "https://test-resource.openai.azure.com/openai/deployments/gpt-4/chat/completions?api-version=2024-10-21",
      },
    });

    await waitFor(() => {
      expect(apiVersionInput).toHaveValue("2024-10-21");
    });

    fireEvent.change(apiBaseInput, {
      target: {
        value: "https://test-resource.openai.azure.com/openai/deployments/gpt-4/chat/completions",
      },
    });

    await waitFor(() => {
      expect(apiVersionInput).toHaveValue("");
    });
  });

  it("preserves a manually edited Azure API version when the API base has no version parameter", async () => {
    const queryClient = createQueryClient();
    render(
      <QueryClientProvider client={queryClient}>
        <MountedFormHost>
          <ProviderSpecificFields selectedProvider={Providers.Azure} />
        </MountedFormHost>
      </QueryClientProvider>,
    );

    const apiBaseInput = await screen.findByPlaceholderText("https://...");
    const apiVersionInput = await screen.findByPlaceholderText("2023-07-01-preview");

    fireEvent.change(apiBaseInput, {
      target: {
        value:
          "https://test-resource.openai.azure.com/openai/deployments/gpt-4/chat/completions?api-version=2024-10-21",
      },
    });

    await waitFor(() => {
      expect(apiVersionInput).toHaveValue("2024-10-21");
    });

    fireEvent.change(apiVersionInput, {
      target: {
        value: "2025-01-01-preview",
      },
    });

    await waitFor(() => {
      expect(apiVersionInput).toHaveValue("2025-01-01-preview");
    });

    fireEvent.change(apiBaseInput, {
      target: {
        value: "https://test-resource.openai.azure.com/openai/deployments/gpt-4/chat/completions",
      },
    });

    await waitFor(() => {
      expect(apiVersionInput).toHaveValue("2025-01-01-preview");
    });
  });

  describe("credential_variants", () => {
    const IdentitySourceProbe = () => {
      const { watch } = useFormContext<MountedFormValues>();
      return <output data-testid="identity-source">{String(watch("anthropic_identity_source") ?? "")}</output>;
    };

    it("defaults to the api_key variant and hides WIF fields", async () => {
      const queryClient = createQueryClient();
      render(
        <QueryClientProvider client={queryClient}>
          <MountedFormHost>
            <ProviderSpecificFields selectedProvider={Providers.Anthropic} />
          </MountedFormHost>
        </QueryClientProvider>,
      );

      expect(await screen.findByLabelText("API Key")).toBeInTheDocument();
      expect(screen.getByLabelText("Upstream API Base")).toBeInTheDocument();
      expect(screen.queryByLabelText("Federation Rule ID")).not.toBeInTheDocument();
      expect(screen.queryByLabelText("Identity Token Reference")).not.toBeInTheDocument();
    });

    it("switching to a WIF variant swaps the rendered fields and unmounts the previous variant's", async () => {
      const queryClient = createQueryClient();
      render(
        <QueryClientProvider client={queryClient}>
          <MountedFormHost>
            <ProviderSpecificFields selectedProvider={Providers.Anthropic} />
          </MountedFormHost>
        </QueryClientProvider>,
      );

      const user = userEvent.setup();
      await screen.findByLabelText("API Key");
      await user.click(await screen.findByRole("combobox", { name: "Authentication method" }));
      await user.click(await screen.findByRole("option", { name: "Workload Identity Federation (external token)" }));

      expect(await screen.findByLabelText("Federation Rule ID")).toBeInTheDocument();
      expect(screen.getByLabelText("Identity Token Reference")).toBeInTheDocument();
      // api_key/api_base belong only to the api_key variant here, so they must be gone.
      expect(screen.queryByLabelText("API Key")).not.toBeInTheDocument();
      expect(screen.queryByLabelText("Upstream API Base")).not.toBeInTheDocument();
    });

    it("injects the fixed discriminator for a variant without rendering a field for it", async () => {
      const queryClient = createQueryClient();
      render(
        <QueryClientProvider client={queryClient}>
          <MountedFormHost>
            <ProviderSpecificFields selectedProvider={Providers.Anthropic} />
            <IdentitySourceProbe />
          </MountedFormHost>
        </QueryClientProvider>,
      );

      const user = userEvent.setup();
      await screen.findByLabelText("API Key");
      expect(screen.getByTestId("identity-source")).toBeEmptyDOMElement();

      await user.click(await screen.findByRole("combobox", { name: "Authentication method" }));
      await user.click(await screen.findByRole("option", { name: "Workload Identity Federation (LiteLLM-signed)" }));

      expect(await screen.findByLabelText("Issuer URL")).toBeInTheDocument();
      expect(screen.queryByLabelText("anthropic_identity_source")).not.toBeInTheDocument();
      await waitFor(() => expect(screen.getByTestId("identity-source")).toHaveTextContent("internal_issuer"));
    });

    it("labels a secret-reference field as a reference, not a raw secret input", async () => {
      const queryClient = createQueryClient();
      render(
        <QueryClientProvider client={queryClient}>
          <MountedFormHost>
            <ProviderSpecificFields selectedProvider={Providers.Anthropic} />
          </MountedFormHost>
        </QueryClientProvider>,
      );

      const user = userEvent.setup();
      await screen.findByLabelText("API Key");
      await user.click(await screen.findByRole("combobox", { name: "Authentication method" }));
      await user.click(await screen.findByRole("option", { name: "Workload Identity Federation (LiteLLM-signed)" }));

      const signingKeyInput = await screen.findByLabelText("Signing Key Reference");
      // A *_ref field renders as plain text, never password -- it must never invite a pasted secret.
      expect(signingKeyInput).toHaveAttribute("type", "text");
    });

    it("infers the wif_token variant is already active when editing a credential that has one", async () => {
      const queryClient = createQueryClient();
      render(
        <QueryClientProvider client={queryClient}>
          <MountedFormHost
            defaultValues={{
              anthropic_federation_rule_id: "rule-1",
              anthropic_organization_id: "org-1",
              anthropic_identity_token: "oidc/env/TOKEN",
            }}
          >
            <ProviderSpecificFields selectedProvider={Providers.Anthropic} />
          </MountedFormHost>
        </QueryClientProvider>,
      );

      expect(await screen.findByLabelText("Identity Token Reference")).toBeInTheDocument();
      expect(screen.queryByLabelText("API Key")).not.toBeInTheDocument();
    });
  });
});
