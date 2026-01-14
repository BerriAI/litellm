import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { Providers } from "../provider_info_helpers";
import AddCredentialModal from "./AddCredentialModal";

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

const mockUploadProps = {
  beforeUpload: vi.fn(),
  onChange: vi.fn(),
};

describe("AddCredentialModal", () => {
  it("should render", () => {
    const queryClient = createQueryClient();
    const onCancel = vi.fn();
    const onAddCredential = vi.fn();

    render(
      <QueryClientProvider client={queryClient}>
        <AddCredentialModal
          open={true}
          onCancel={onCancel}
          onAddCredential={onAddCredential}
          uploadProps={mockUploadProps}
        />
      </QueryClientProvider>,
    );

    expect(screen.getByText("Add New Credential")).toBeInTheDocument();
    expect(screen.getByLabelText("Credential Name:")).toBeInTheDocument();
    expect(screen.getByLabelText("Provider:")).toBeInTheDocument();
  });

  it("should show the correct provider fields", async () => {
    const queryClient = createQueryClient();
    const onCancel = vi.fn();
    const onAddCredential = vi.fn();

    render(
      <QueryClientProvider client={queryClient}>
        <AddCredentialModal
          open={true}
          onCancel={onCancel}
          onAddCredential={onAddCredential}
          uploadProps={mockUploadProps}
        />
      </QueryClientProvider>,
    );

    await waitFor(() => {
      expect(screen.getByLabelText("OpenAI API Key")).toBeInTheDocument();
      expect(screen.getByPlaceholderText("https://api.openai.com/v1")).toBeInTheDocument();
    });
  });
});
