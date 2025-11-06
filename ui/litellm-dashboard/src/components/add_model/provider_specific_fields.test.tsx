import { render, waitFor } from "@testing-library/react";
import { describe, it, expect, beforeAll } from "vitest";
import { Form } from "antd";
import { Providers } from "../provider_info_helpers";
import ProviderSpecificFields from "./provider_specific_fields";

// Mock window.matchMedia for Ant Design components
beforeAll(() => {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: (query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: () => {}, // deprecated
      removeListener: () => {}, // deprecated
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => false,
    }),
  });
});

describe("ProviderSpecificFields", () => {
  it("should render the provider specific fields for OpenAI", async () => {
    const { getByLabelText, getByPlaceholderText } = render(
      <Form>
        <ProviderSpecificFields selectedProvider={Providers.OpenAI} />
      </Form>,
    );

    await waitFor(() => {
      // Check for the API Base text input
      const apiBaseInput = getByPlaceholderText("https://api.openai.com/v1");
      expect(apiBaseInput).toBeInTheDocument();
      expect(apiBaseInput).toHaveAttribute("type", "text");

      // Check for Organization field
      const orgInput = getByPlaceholderText("[OPTIONAL] my-unique-org");
      expect(orgInput).toBeInTheDocument();

      // Check for API Key field
      const apiKeyLabel = getByLabelText("OpenAI API Key");
      expect(apiKeyLabel).toBeInTheDocument();
    });
  });

  it("should render the provider specific fields for vLLM", async () => {
    const { getByLabelText, getByPlaceholderText } = render(
      <Form>
        <ProviderSpecificFields selectedProvider={"Hosted_Vllm" as Providers} />
      </Form>,
    );

    await waitFor(() => {
      const apiBaseInput = getByPlaceholderText("https://...");
      expect(apiBaseInput).toBeInTheDocument();
      expect(apiBaseInput).toHaveAttribute("type", "text");

      // Check for API Key field
      const apiKeyLabel = getByLabelText("vLLM API Key");
      expect(apiKeyLabel).toBeInTheDocument();
    });
  });

  it("should render the provider specific fields for Azure", async () => {
    const { getByLabelText, getByPlaceholderText } = render(
      <Form>
        <ProviderSpecificFields selectedProvider={Providers.Azure} />
      </Form>,
    );

    await waitFor(() => {
      // Check for API Base field
      const apiBaseInput = getByPlaceholderText("https://...");
      expect(apiBaseInput).toBeInTheDocument();
      expect(apiBaseInput).toHaveAttribute("type", "text");

      // Check for API Version field
      const apiVersionInput = getByPlaceholderText("2023-07-01-preview");
      expect(apiVersionInput).toBeInTheDocument();

      // Check for Base Model field
      const baseModelInput = getByPlaceholderText("azure/gpt-3.5-turbo");
      expect(baseModelInput).toBeInTheDocument();

      // Check for API Key field
      const apiKeyInput = getByLabelText("Azure API Key");
      expect(apiKeyInput).toBeInTheDocument();
      expect(apiKeyInput).toHaveAttribute("type", "password");
      expect(apiKeyInput).toHaveAttribute("placeholder", "Enter your Azure API Key");

      // Check for Azure AD Token field
      const azureAdTokenInput = getByLabelText("Azure AD Token");
      expect(azureAdTokenInput).toBeInTheDocument();
      expect(azureAdTokenInput).toHaveAttribute("type", "password");
      expect(azureAdTokenInput).toHaveAttribute("placeholder", "Enter your Azure AD Token");
    });
  });
});
