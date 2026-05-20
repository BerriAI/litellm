import { describe, it, expect } from "vitest";
import {
  client,
  createClient,
  createConfig,
  chatCompletionV1ChatCompletionsPost,
  embeddingsV1EmbeddingsPost,
  responsesApiV1ResponsesPost,
  ocrV1OcrPost,
  rerankV1RerankPost,
  imageGenerationV1ImagesGenerationsPost,
  moderationsV1ModerationsPost,
  audioSpeechV1AudioSpeechPost,
  audioTranscriptionsV1AudioTranscriptionsPost,
  completionV1CompletionsPost,
  generateKeyFnKeyGeneratePost,
  listKeysKeyListGet,
  modelListV1ModelsGet,
  healthEndpointHealthGet,
  searchV1SearchPost,
} from "./index";

describe("SDK exports", () => {
  it("should export the default client", () => {
    expect(client).toBeDefined();
    expect(typeof client.get).toBe("function");
    expect(typeof client.post).toBe("function");
    expect(typeof client.setConfig).toBe("function");
  });

  it("should export createClient and createConfig", () => {
    expect(typeof createClient).toBe("function");
    expect(typeof createConfig).toBe("function");
  });

  it("should export core LLM API functions", () => {
    expect(typeof chatCompletionV1ChatCompletionsPost).toBe("function");
    expect(typeof embeddingsV1EmbeddingsPost).toBe("function");
    expect(typeof responsesApiV1ResponsesPost).toBe("function");
    expect(typeof ocrV1OcrPost).toBe("function");
    expect(typeof rerankV1RerankPost).toBe("function");
    expect(typeof imageGenerationV1ImagesGenerationsPost).toBe("function");
    expect(typeof moderationsV1ModerationsPost).toBe("function");
    expect(typeof audioSpeechV1AudioSpeechPost).toBe("function");
    expect(typeof audioTranscriptionsV1AudioTranscriptionsPost).toBe(
      "function"
    );
    expect(typeof completionV1CompletionsPost).toBe("function");
    expect(typeof searchV1SearchPost).toBe("function");
  });

  it("should export management API functions", () => {
    expect(typeof generateKeyFnKeyGeneratePost).toBe("function");
    expect(typeof listKeysKeyListGet).toBe("function");
    expect(typeof modelListV1ModelsGet).toBe("function");
    expect(typeof healthEndpointHealthGet).toBe("function");
  });
});

describe("Client configuration", () => {
  it("should create a new client instance with custom config", () => {
    const customClient = createClient(
      createConfig({
        baseUrl: "http://localhost:4000",
      })
    );
    expect(customClient).toBeDefined();
    expect(typeof customClient.get).toBe("function");
    expect(typeof customClient.post).toBe("function");
  });

  it("should allow setting config on the default client", () => {
    expect(() => {
      client.setConfig({
        baseUrl: "http://localhost:4000",
      });
    }).not.toThrow();
  });

  it("should allow setting config with auth headers", () => {
    expect(() => {
      client.setConfig({
        baseUrl: "http://localhost:4000",
        headers: {
          Authorization: "Bearer sk-test-key",
        },
      });
    }).not.toThrow();
  });
});
