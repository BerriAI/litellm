import { describe, it, expect } from "vitest";
import { generateCodeSnippet } from "./CodeSnippets";
import { EndpointType } from "./mode_endpoint_mapping";

describe("CodeSnippets", () => {
  it("should generate the correct code snippet for embeddings", () => {
    const code = generateCodeSnippet({
      endpointType: EndpointType.EMBEDDINGS,
      inputMessage: "Hello, world!",
      selectedModel: "text-embedding-3-small",
      apiKeySource: "session",
      accessToken: "1234567890",
      apiKey: "1234567890",
      chatHistory: [],
      selectedTags: [],
      selectedVectorStores: [],
      selectedGuardrails: [],
      selectedMCPTools: [],
      selectedSdk: "openai",
      selectedVoice: "alloy",
    });
    expect(code).toContain("text-embedding-3-small");
    expect(code).toContain("Hello, world!");
    expect(code).toContain("client.embeddings.create");
    expect(code).toContain("print(response.data[0].embedding)");
  });
});
