import { describe, expect, it } from "vitest";
import { buildSearchToolPayload } from "./searchToolPayload";

const minimal = { search_tool_name: "tool", search_provider: "perplexity" };

describe("buildSearchToolPayload", () => {
  it("nests the provider under litellm_params", () => {
    expect(buildSearchToolPayload(minimal).litellm_params.search_provider).toBe("perplexity");
  });

  it("leaves every optional param undefined so it never reaches the wire body", () => {
    expect(JSON.stringify(buildSearchToolPayload(minimal))).toBe(
      '{"search_tool_name":"tool","litellm_params":{"search_provider":"perplexity"}}',
    );
  });

  it("keeps the full key set in the object even when the optional params are absent", () => {
    expect(buildSearchToolPayload(minimal)).toStrictEqual({
      search_tool_name: "tool",
      litellm_params: {
        search_provider: "perplexity",
        api_key: undefined,
        api_base: undefined,
        timeout: undefined,
        max_retries: undefined,
      },
      search_tool_info: undefined,
    });
  });

  it("passes an api key through untouched", () => {
    expect(buildSearchToolPayload({ ...minimal, api_key: "sk-secret" }).litellm_params.api_key).toBe("sk-secret");
  });

  it("keeps an explicitly emptied api key as an empty string, matching the antd store", () => {
    expect(buildSearchToolPayload({ ...minimal, api_key: "" }).litellm_params.api_key).toBe("");
  });

  it("wraps a description in search_tool_info", () => {
    expect(buildSearchToolPayload({ ...minimal, description: "finds things" }).search_tool_info).toStrictEqual({
      description: "finds things",
    });
  });

  it("drops search_tool_info for an empty description", () => {
    expect(buildSearchToolPayload({ ...minimal, description: "" }).search_tool_info).toBeUndefined();
  });

  it("parses timeout as a float", () => {
    expect(buildSearchToolPayload({ ...minimal, timeout: "2.5" }).litellm_params.timeout).toBe(2.5);
  });

  it("parses max_retries as an integer and truncates a decimal", () => {
    expect(buildSearchToolPayload({ ...minimal, max_retries: "3.9" }).litellm_params.max_retries).toBe(3);
  });

  it('parses a "0" timeout as 0, because the original guard tests the string not the number', () => {
    expect(buildSearchToolPayload({ ...minimal, timeout: "0" }).litellm_params.timeout).toBe(0);
  });

  it("treats an empty timeout string as absent", () => {
    expect(buildSearchToolPayload({ ...minimal, timeout: "" }).litellm_params.timeout).toBeUndefined();
  });
});
