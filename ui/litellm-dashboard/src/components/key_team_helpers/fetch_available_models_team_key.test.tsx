import { describe, expect, it } from "vitest";

import { getModelDisplayName, unfurlWildcardModelsInList } from "./fetch_available_models_team_key";

describe("getModelDisplayName", () => {
  it("should return display label for all proxy models", () => {
    expect(getModelDisplayName("all-proxy-models")).toBe("All Proxy Models");
  });

  it("should return provider-wide label for wildcard models", () => {
    expect(getModelDisplayName("openai/*")).toBe("All openai models");
  });
});

describe("unfurlWildcardModelsInList", () => {
  it("should expand wildcard models while preserving display names and removing duplicates", () => {
    expect(
      unfurlWildcardModelsInList(
        ["openai/*", "anthropic/claude-4", "openai/gpt-4"],
        ["openai/gpt-4", "openai/gpt-4o", "anthropic/claude-4"],
      ),
    ).toEqual(["openai/*", "openai/gpt-4", "openai/gpt-4o", "anthropic/claude-4"]);
  });

  it("should not exceed the call stack when expanding a wildcard with many matching models", () => {
    const allModels = Array.from({ length: 150_000 }, (_, index) => `openai/model-${index}`);

    const result = unfurlWildcardModelsInList(["openai/*"], allModels);

    expect(result).toHaveLength(allModels.length + 1);
    expect(result[0]).toBe("openai/*");
    expect(result[1]).toBe("openai/model-0");
    expect(result.at(-1)).toBe("openai/model-149999");
  });
});
