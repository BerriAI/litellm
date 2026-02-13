import { describe, expect, it } from "vitest";
import {
  buildCallbackPayload,
  parseEnabledProviders,
  reconstructCallbacksList,
} from "./callbackPayloadHelpers";

describe("parseEnabledProviders", () => {
  it("should parse comma-separated string into trimmed array", () => {
    expect(parseEnabledProviders("bedrock, azure , vertex_ai")).toEqual([
      "bedrock",
      "azure",
      "vertex_ai",
    ]);
  });

  it("should return array as-is (trimmed)", () => {
    expect(parseEnabledProviders(["bedrock", " azure "])).toEqual(["bedrock", "azure"]);
  });

  it("should filter empty strings", () => {
    expect(parseEnabledProviders("bedrock,,azure")).toEqual(["bedrock", "azure"]);
  });

  it("should return empty array for null/undefined/non-string", () => {
    expect(parseEnabledProviders(null)).toEqual([]);
    expect(parseEnabledProviders(undefined)).toEqual([]);
    expect(parseEnabledProviders(1)).toEqual([]);
  });

  it("should return empty array for empty string", () => {
    expect(parseEnabledProviders("")).toEqual([]);
  });
});

describe("reconstructCallbacksList", () => {
  const successAndFailure = (name: string, params?: Record<string, unknown>) =>
    ({ name, type: "success_and_failure", ...(params && { params }) });

  it("should include only success_and_failure callbacks", () => {
    const list = reconstructCallbacksList([
      successAndFailure("prometheus"),
      { name: "langfuse", type: "success" },
    ]);
    expect(list).toEqual(["prometheus"]);
  });

  it("should convert callbacks with params to dict-style entry", () => {
    const list = reconstructCallbacksList([
      successAndFailure("websearch_interception", { enabled_providers: ["bedrock"] }),
    ]);
    expect(list).toEqual([{ websearch_interception: { enabled_providers: ["bedrock"] } }]);
  });

  it("should mix string and dict-style entries", () => {
    const list = reconstructCallbacksList([
      successAndFailure("prometheus"),
      successAndFailure("websearch_interception", { enabled_providers: ["azure"] }),
    ]);
    expect(list).toEqual(["prometheus", { websearch_interception: { enabled_providers: ["azure"] } }]);
  });

  it("should return empty array when list has only success (or failure) type", () => {
    expect(reconstructCallbacksList([{ name: "langfuse", type: "success" }])).toEqual([]);
  });
});

describe("buildCallbackPayload", () => {
  const getCallbacks = (payload: Record<string, unknown>) =>
    (payload.litellm_settings as { callbacks: unknown[] }).callbacks;

  const existingWithWebsearch = [
    { name: "prometheus", type: "success_and_failure" as const },
    {
      name: "websearch_interception",
      type: "success_and_failure" as const,
      params: { enabled_providers: ["bedrock"] },
    },
  ];

  describe("websearch_interception", () => {
    it("should build litellm_settings.callbacks when adding", () => {
      const payload = buildCallbackPayload(
        { enabled_providers: "bedrock, azure", search_tool_name: "perplexity-search" },
        "websearch_interception",
        [],
        false
      );
      expect(payload).toEqual({
        litellm_settings: {
          callbacks: [
            {
              websearch_interception: {
                enabled_providers: ["bedrock", "azure"],
                search_tool_name: "perplexity-search",
              },
            },
          ],
        },
      });
    });

    it("should merge with existing callbacks when adding", () => {
      const payload = buildCallbackPayload(
        { enabled_providers: "vertex_ai" },
        "websearch_interception",
        existingWithWebsearch,
        false
      );
      const callbacks = getCallbacks(payload);
      expect(callbacks).toHaveLength(3);
      expect(callbacks[0]).toBe("prometheus");
      expect(callbacks[1]).toEqual({ websearch_interception: { enabled_providers: ["bedrock"] } });
      expect(callbacks[2]).toEqual({ websearch_interception: { enabled_providers: ["vertex_ai"] } });
    });

    it("should replace websearch_interception entry when editing", () => {
      const payload = buildCallbackPayload(
        { enabled_providers: "bedrock, azure, vertex_ai", search_tool_name: "" },
        "websearch_interception",
        existingWithWebsearch,
        true
      );
      const callbacks = getCallbacks(payload);
      expect(callbacks).toHaveLength(2);
      expect(callbacks[0]).toBe("prometheus");
      expect(callbacks[1]).toEqual({
        websearch_interception: { enabled_providers: ["bedrock", "azure", "vertex_ai"] },
      });
    });

    it("should omit search_tool_name when empty or whitespace", () => {
      const payload = buildCallbackPayload(
        { enabled_providers: "bedrock", search_tool_name: "  " },
        "websearch_interception",
        undefined,
        false
      );
      const entry = getCallbacks(payload)[0] as Record<string, unknown>;
      expect(entry.websearch_interception).toEqual({ enabled_providers: ["bedrock"] });
      expect((entry.websearch_interception as Record<string, unknown>).search_tool_name).toBeUndefined();
    });

    it("should build payload with empty enabled_providers when none given", () => {
      const payload = buildCallbackPayload(
        { enabled_providers: "", search_tool_name: "" },
        "websearch_interception",
        [],
        false
      );
      expect(payload).toEqual({
        litellm_settings: {
          callbacks: [{ websearch_interception: { enabled_providers: [] } }],
        },
      });
    });
  });

  describe("non-websearch callbacks", () => {
    it("should build environment_variables and success_callback", () => {
      const payload = buildCallbackPayload(
        { LANGFUSE_PUBLIC_KEY: "pk", LANGFUSE_SECRET_KEY: "sk" },
        "langfuse",
        undefined,
        false
      );
      expect(payload).toEqual({
        environment_variables: { LANGFUSE_PUBLIC_KEY: "pk", LANGFUSE_SECRET_KEY: "sk" },
        litellm_settings: { success_callback: ["langfuse"] },
      });
    });
  });
});
