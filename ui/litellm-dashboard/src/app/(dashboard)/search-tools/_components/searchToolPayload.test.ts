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

  it("builds only the params a form actually collects", () => {
    expect(buildSearchToolPayload(minimal)).toStrictEqual({
      search_tool_name: "tool",
      litellm_params: {
        search_provider: "perplexity",
        api_key: undefined,
      },
      search_tool_info: undefined,
    });
  });

  it("passes an api key through untouched", () => {
    expect(buildSearchToolPayload({ ...minimal, api_key: "sk-secret" }).litellm_params.api_key).toBe("sk-secret");
  });

  it("keeps an explicitly emptied api key as an empty string, so the backend clears it", () => {
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

  it("sends the same wire body with an api key and a description as it did before the params were pruned", () => {
    expect(
      JSON.stringify(buildSearchToolPayload({ ...minimal, api_key: "sk-secret", description: "finds things" })),
    ).toBe(
      '{"search_tool_name":"tool","litellm_params":{"search_provider":"perplexity","api_key":"sk-secret"},"search_tool_info":{"description":"finds things"}}',
    );
  });
});
