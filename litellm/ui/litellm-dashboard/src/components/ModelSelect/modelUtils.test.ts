import { describe, expect, it } from "vitest";
import { splitWildcardModels } from "./modelUtils";

describe("splitWildcardModels", () => {
  it("should return empty arrays when given empty array", () => {
    const result = splitWildcardModels([]);
    expect(result).toEqual({ wildcard: [], regular: [] });
  });

  it("should split models into wildcard and regular groups", () => {
    const models: string[] = ["gpt-4", "openai/*", "claude-3", "anthropic/*"];

    const result = splitWildcardModels(models);

    expect(result.wildcard).toHaveLength(2);
    expect(result.wildcard[0]).toBe("openai/*");
    expect(result.wildcard[1]).toBe("anthropic/*");

    expect(result.regular).toHaveLength(2);
    expect(result.regular[0]).toBe("gpt-4");
    expect(result.regular[1]).toBe("claude-3");
  });

  it("should return only wildcard models when all models are wildcard", () => {
    const models: string[] = ["openai/*", "anthropic/*"];

    const result = splitWildcardModels(models);

    expect(result.wildcard).toHaveLength(2);
    expect(result.regular).toHaveLength(0);
  });

  it("should return only regular models when no models are wildcard", () => {
    const models: string[] = ["gpt-4", "claude-3"];

    const result = splitWildcardModels(models);

    expect(result.wildcard).toHaveLength(0);
    expect(result.regular).toHaveLength(2);
  });

  it("should correctly identify wildcard models ending with /*", () => {
    const models: string[] = ["provider/*", "not-wildcard", "also-not/*/wildcard"];

    const result = splitWildcardModels(models);

    expect(result.wildcard).toHaveLength(1);
    expect(result.wildcard[0]).toBe("provider/*");
    expect(result.regular).toHaveLength(2);
  });
});
