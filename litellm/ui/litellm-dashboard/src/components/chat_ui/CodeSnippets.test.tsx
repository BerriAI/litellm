import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { generateCodeSnippet } from "./CodeSnippets";
import { EndpointType } from "./mode_endpoint_mapping";

describe("CodeSnippets", () => {
  const originalLocation = window.location;

  beforeEach(() => {
    // Mock window.location.origin
    Object.defineProperty(window, "location", {
      value: {
        ...originalLocation,
        origin: "http://localhost:4000",
      },
      writable: true,
    });
  });

  afterEach(() => {
    Object.defineProperty(window, "location", {
      value: originalLocation,
      writable: true,
    });
  });

  const baseParams = {
    endpointType: EndpointType.EMBEDDINGS,
    inputMessage: "Hello, world!",
    selectedModel: "text-embedding-3-small",
    apiKeySource: "session" as const,
    accessToken: "1234567890",
    apiKey: "1234567890",
    chatHistory: [],
    selectedTags: [],
    selectedVectorStores: [],
    selectedGuardrails: [],
    selectedPolicies: [],
    selectedMCPServers: [],
    selectedSdk: "openai" as const,
    selectedVoice: "alloy",
  };

  it("should generate the correct code snippet for embeddings", () => {
    const code = generateCodeSnippet(baseParams);
    expect(code).toContain("text-embedding-3-small");
    expect(code).toContain("Hello, world!");
    expect(code).toContain("client.embeddings.create");
    expect(code).toContain("print(response.data[0].embedding)");
  });

  describe("custom headers", () => {
    const customHeaders = { "anthropic-beta": "context-1m-2025-08-07", "x-request-source": "playground" };

    it("passes configured headers as default_headers on the OpenAI client", () => {
      const code = generateCodeSnippet({ ...baseParams, endpointType: EndpointType.CHAT, customHeaders });
      expect(code).toContain('base_url="http://localhost:4000",\n\tdefault_headers={');
      expect(code).toContain('"anthropic-beta": "context-1m-2025-08-07"');
      expect(code).toContain('"x-request-source": "playground"');
    });

    it("passes configured headers as default_headers on the Azure client", () => {
      const code = generateCodeSnippet({ ...baseParams, selectedSdk: "azure", customHeaders });
      expect(code).toContain('api_version="2024-02-01",\n\tdefault_headers={');
      expect(code).toContain('"anthropic-beta": "context-1m-2025-08-07"');
    });

    it("omits default_headers when no custom headers are configured", () => {
      expect(generateCodeSnippet(baseParams)).not.toContain("default_headers");
      expect(generateCodeSnippet({ ...baseParams, customHeaders: {} })).not.toContain("default_headers");
    });
  });

  describe("base URL selection", () => {
    it("should use LITELLM_UI_API_DOC_BASE_URL when provided", () => {
      const customBaseUrl = "https://custom-doc.example.com";
      const code = generateCodeSnippet({
        ...baseParams,
        proxySettings: {
          LITELLM_UI_API_DOC_BASE_URL: customBaseUrl,
        },
      });
      expect(code).toContain(`base_url="${customBaseUrl}"`);
    });

    it("should use PROXY_BASE_URL when LITELLM_UI_API_DOC_BASE_URL is not provided", () => {
      const proxyBaseUrl = "https://proxy.example.com";
      const code = generateCodeSnippet({
        ...baseParams,
        proxySettings: {
          PROXY_BASE_URL: proxyBaseUrl,
        },
      });
      expect(code).toContain(`base_url="${proxyBaseUrl}"`);
    });

    it("should prioritize LITELLM_UI_API_DOC_BASE_URL over PROXY_BASE_URL when both are provided", () => {
      const customBaseUrl = "https://custom-doc.example.com";
      const proxyBaseUrl = "https://proxy.example.com";
      const code = generateCodeSnippet({
        ...baseParams,
        proxySettings: {
          LITELLM_UI_API_DOC_BASE_URL: customBaseUrl,
          PROXY_BASE_URL: proxyBaseUrl,
        },
      });
      expect(code).toContain(`base_url="${customBaseUrl}"`);
      expect(code).not.toContain(`base_url="${proxyBaseUrl}"`);
    });

    it("should fallback to window.location.origin when proxySettings is not provided", () => {
      const code = generateCodeSnippet(baseParams);
      expect(code).toContain(`base_url="http://localhost:4000"`);
    });

    it("should fallback to window.location.origin when proxySettings is empty", () => {
      const code = generateCodeSnippet({
        ...baseParams,
        proxySettings: {},
      });
      expect(code).toContain(`base_url="http://localhost:4000"`);
    });

    it("should fallback to PROXY_BASE_URL when LITELLM_UI_API_DOC_BASE_URL is empty string", () => {
      const proxyBaseUrl = "https://proxy.example.com";
      const code = generateCodeSnippet({
        ...baseParams,
        proxySettings: {
          LITELLM_UI_API_DOC_BASE_URL: "",
          PROXY_BASE_URL: proxyBaseUrl,
        },
      });
      expect(code).toContain(`base_url="${proxyBaseUrl}"`);
    });

    it("should fallback to PROXY_BASE_URL when LITELLM_UI_API_DOC_BASE_URL is whitespace only", () => {
      const proxyBaseUrl = "https://proxy.example.com";
      const code = generateCodeSnippet({
        ...baseParams,
        proxySettings: {
          LITELLM_UI_API_DOC_BASE_URL: "   ",
          PROXY_BASE_URL: proxyBaseUrl,
        },
      });
      expect(code).toContain(`base_url="${proxyBaseUrl}"`);
    });

    it("should fallback to window.location.origin when LITELLM_UI_API_DOC_BASE_URL is null", () => {
      const code = generateCodeSnippet({
        ...baseParams,
        proxySettings: {
          LITELLM_UI_API_DOC_BASE_URL: null,
        },
      });
      expect(code).toContain(`base_url="http://localhost:4000"`);
    });

    it("should use LITELLM_UI_API_DOC_BASE_URL for Azure SDK", () => {
      const customBaseUrl = "https://custom-doc.example.com";
      const code = generateCodeSnippet({
        ...baseParams,
        selectedSdk: "azure",
        proxySettings: {
          LITELLM_UI_API_DOC_BASE_URL: customBaseUrl,
        },
      });
      expect(code).toContain(`azure_endpoint="${customBaseUrl}"`);
    });

    it("should use PROXY_BASE_URL for Azure SDK when LITELLM_UI_API_DOC_BASE_URL is not provided", () => {
      const proxyBaseUrl = "https://proxy.example.com";
      const code = generateCodeSnippet({
        ...baseParams,
        selectedSdk: "azure",
        proxySettings: {
          PROXY_BASE_URL: proxyBaseUrl,
        },
      });
      expect(code).toContain(`azure_endpoint="${proxyBaseUrl}"`);
    });
  });
});
