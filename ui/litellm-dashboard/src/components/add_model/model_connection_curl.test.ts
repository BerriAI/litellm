import { describe, expect, it } from "vitest";
import { buildRequestUrl, buildTestConnectionCurl } from "./model_connection_curl";

describe("buildRequestUrl", () => {
  it("appends the chat endpoint path to the base url", () => {
    expect(buildRequestUrl("https://inference.poolside.ai/v1", "chat")).toBe(
      "https://inference.poolside.ai/v1/chat/completions",
    );
  });

  it("normalizes a trailing slash before appending the endpoint", () => {
    expect(buildRequestUrl("https://inference.poolside.ai/v1/", "chat")).toBe(
      "https://inference.poolside.ai/v1/chat/completions",
    );
  });

  it("uses the embedding endpoint for embedding mode", () => {
    expect(buildRequestUrl("https://api.example.com/v1", "embedding")).toBe("https://api.example.com/v1/embeddings");
  });

  it("does not double-append when the base already ends with the endpoint", () => {
    expect(buildRequestUrl("https://api.example.com/v1/chat/completions", "chat")).toBe(
      "https://api.example.com/v1/chat/completions",
    );
  });

  it("returns the base unchanged for an unknown mode", () => {
    expect(buildRequestUrl("https://api.example.com/v1", "mystery")).toBe("https://api.example.com/v1");
  });
});

describe("buildTestConnectionCurl", () => {
  it("targets the full endpoint url, not the bare base", () => {
    const curl = buildTestConnectionCurl({
      apiBase: "https://inference.poolside.ai/v1/",
      testMode: "chat",
      requestBody: { model: "poolside/laguna-s-2.1", messages: [] },
      requestHeaders: { Authorization: "Bearer sk-xxx" },
    });

    expect(curl).toContain("https://inference.poolside.ai/v1/chat/completions");
    expect(curl).not.toContain("https://inference.poolside.ai/v1/ \\");
  });

  it("produces a single valid JSON body without wrapping braces", () => {
    const requestBody = { model: "poolside/laguna-s-2.1", max_tokens: 16 };
    const curl = buildTestConnectionCurl({
      apiBase: "https://inference.poolside.ai/v1",
      testMode: "chat",
      requestBody,
      requestHeaders: {},
    });

    const jsonPart = curl.slice(curl.indexOf("-d '") + 4, curl.lastIndexOf("'"));
    expect(() => JSON.parse(jsonPart)).not.toThrow();
    expect(JSON.parse(jsonPart)).toEqual(requestBody);
    // the old bug wrapped the object in an extra brace pair: "{\n  {"
    expect(curl).not.toMatch(/-d '\{\s*\{/);
  });

  it("includes provided headers", () => {
    const curl = buildTestConnectionCurl({
      apiBase: "https://api.example.com/v1",
      testMode: "chat",
      requestBody: { model: "m" },
      requestHeaders: { Authorization: "Bearer sk-xxx" },
    });

    expect(curl).toContain("-H 'Authorization: Bearer sk-xxx'");
    expect(curl).toContain("-H 'Content-Type: application/json'");
  });
});
