import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
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
});
