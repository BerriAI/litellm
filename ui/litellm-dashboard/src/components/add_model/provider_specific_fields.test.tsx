import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import { Form } from "antd";
import { beforeAll, describe, expect, it, vi } from "vitest";
import { Providers } from "../provider_info_helpers";
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

describe("ProviderSpecificFields", () => {
  it("should render", async () => {
    const queryClient = createQueryClient();
    render(
      <QueryClientProvider client={queryClient}>
        <Form>
          <ProviderSpecificFields selectedProvider={Providers.OpenAI} />
        </Form>
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
        <Form>
          <ProviderSpecificFields selectedProvider={Providers.OpenAI} />
        </Form>
      </QueryClientProvider>,
    );

    await waitFor(() => {
      const apiKeyLabel = screen.getByLabelText("OpenAI API Key");
      expect(apiKeyLabel).toBeInTheDocument();

      const apiBaseInput = screen.getByPlaceholderText("https://api.openai.com/v1");
      expect(apiBaseInput).toBeInTheDocument();
      expect(apiBaseInput).toHaveAttribute("type", "text");

      const orgInput = screen.getByPlaceholderText("[OPTIONAL] my-unique-org");
      expect(orgInput).toBeInTheDocument();
    });
  });

  it("should render the provider specific fields for vLLM", async () => {
    const queryClient = createQueryClient();
    render(
      <QueryClientProvider client={queryClient}>
        <Form>
          <ProviderSpecificFields selectedProvider={"Hosted_Vllm" as Providers} />
        </Form>
      </QueryClientProvider>,
    );

    await waitFor(() => {
      const apiKeyLabel = screen.getByLabelText("vLLM API Key");
      expect(apiKeyLabel).toBeInTheDocument();

      const apiBaseInput = screen.getByPlaceholderText("https://...");
      expect(apiBaseInput).toBeInTheDocument();
      expect(apiBaseInput).toHaveAttribute("type", "text");
    });
  });

  it("should render the provider specific fields for Azure", async () => {
    const queryClient = createQueryClient();
    render(
      <QueryClientProvider client={queryClient}>
        <Form>
          <ProviderSpecificFields selectedProvider={Providers.Azure} />
        </Form>
      </QueryClientProvider>,
    );

    await waitFor(() => {
      const apiKeyInput = screen.getByLabelText("Azure API Key");
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
  });
});
