import { describe, it, expect } from "vitest";
import { splitWildcardModels, type GroupedModels } from "./modelUtils";
import type { ProxyModel } from "@/app/(dashboard)/hooks/models/useModels";

describe("splitWildcardModels", () => {
  it("should return empty arrays when given empty array", () => {
    const result = splitWildcardModels([]);
    expect(result).toEqual({ wildcard: [], regular: [] });
  });

  it("should split models into wildcard and regular groups", () => {
    const models: ProxyModel[] = [
      { id: "gpt-4", object: "model", created: 1234567890, owned_by: "openai" },
      { id: "openai/*", object: "model", created: 1234567890, owned_by: "openai" },
      { id: "claude-3", object: "model", created: 1234567890, owned_by: "anthropic" },
      { id: "anthropic/*", object: "model", created: 1234567890, owned_by: "anthropic" },
    ];

    const result = splitWildcardModels(models);

    expect(result.wildcard).toHaveLength(2);
    expect(result.wildcard[0].id).toBe("openai/*");
    expect(result.wildcard[1].id).toBe("anthropic/*");

    expect(result.regular).toHaveLength(2);
    expect(result.regular[0].id).toBe("gpt-4");
    expect(result.regular[1].id).toBe("claude-3");
  });

  it("should return only wildcard models when all models are wildcard", () => {
    const models: ProxyModel[] = [
      { id: "openai/*", object: "model", created: 1234567890, owned_by: "openai" },
      { id: "anthropic/*", object: "model", created: 1234567890, owned_by: "anthropic" },
    ];

    const result = splitWildcardModels(models);

    expect(result.wildcard).toHaveLength(2);
    expect(result.regular).toHaveLength(0);
  });

  it("should return only regular models when no models are wildcard", () => {
    const models: ProxyModel[] = [
      { id: "gpt-4", object: "model", created: 1234567890, owned_by: "openai" },
      { id: "claude-3", object: "model", created: 1234567890, owned_by: "anthropic" },
    ];

    const result = splitWildcardModels(models);

    expect(result.wildcard).toHaveLength(0);
    expect(result.regular).toHaveLength(2);
  });

  it("should correctly identify wildcard models ending with /*", () => {
    const models: ProxyModel[] = [
      { id: "provider/*", object: "model", created: 1234567890, owned_by: "provider" },
      { id: "not-wildcard", object: "model", created: 1234567890, owned_by: "provider" },
      { id: "also-not/*/wildcard", object: "model", created: 1234567890, owned_by: "provider" },
    ];

    const result = splitWildcardModels(models);

    expect(result.wildcard).toHaveLength(1);
    expect(result.wildcard[0].id).toBe("provider/*");
    expect(result.regular).toHaveLength(2);
  });
});
