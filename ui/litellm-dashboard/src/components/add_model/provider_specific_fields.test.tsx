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
});
