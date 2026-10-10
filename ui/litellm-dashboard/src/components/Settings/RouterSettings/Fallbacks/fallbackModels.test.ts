import { describe, expect, it } from "vitest";
import { buildFallbackEntries, primaryModels, providerModels } from "./fallbackModels";

const models = [
  { model_group: "writer", providers: ["openai"] },
  { model_group: "fast", providers: ["openai"] },
  { model_group: "backup", providers: ["anthropic"] },
  { model_group: "openai/*", providers: ["openai"] },
  { model_group: "unknown" },
];

describe("provider fallback expansion", () => {
  it("uses provider metadata for aliases, deduplicates and excludes wildcard routes", () => {
    expect(providerModels([...models, models[0]], "openai")).toEqual(["fast", "writer"]);
    expect(providerModels(models, "missing")).toEqual([]);
    expect(primaryModels({ id: "1", primaryModel: "unknown", fallbackModels: [] }, models)).toEqual(["unknown"]);
  });

  it("creates concrete router entries and preserves the selected target order", () => {
    expect(
      buildFallbackEntries(
        [
          {
            id: "1",
            primaryModel: null,
            primaryProvider: "openai",
            fallbackModels: ["backup", "other", "backup"],
          },
        ],
        models,
        [{ untouched: ["existing"] }],
      ),
    ).toEqual({
      entries: [{ fast: ["backup", "other"] }, { writer: ["backup", "other"] }],
    });
  });

  it("rejects provider selections that overlap existing chains or another group", () => {
    const group = { id: "1", primaryModel: null, primaryProvider: "openai", fallbackModels: ["backup"] };
    expect(buildFallbackEntries([group], models, [{ writer: ["existing"] }]).error).toContain("writer");
    expect(
      buildFallbackEntries([group, { id: "2", primaryModel: "fast", fallbackModels: ["backup"] }], models, []).error,
    ).toContain("fast");
  });

  it("rejects empty provider groups and chains without a distinct fallback", () => {
    expect(
      buildFallbackEntries(
        [
          {
            id: "1",
            primaryModel: null,
            primaryProvider: "missing",
            fallbackModels: ["backup"],
          },
        ],
        models,
        [],
      ).error,
    ).toContain("complete configuration");
    expect(
      buildFallbackEntries(
        [
          {
            id: "1",
            primaryModel: "fast",
            fallbackModels: ["fast"],
          },
        ],
        models,
        [],
      ).error,
    ).toContain("different");
  });

  it("removes self-fallbacks without changing the remaining order", () => {
    expect(
      buildFallbackEntries(
        [
          {
            id: "1",
            primaryModel: "fast",
            fallbackModels: ["backup", "fast", "other"],
          },
        ],
        models,
        [],
      ),
    ).toEqual({ entries: [{ fast: ["backup", "other"] }] });
  });

  it("rejects excessive chains instead of silently dropping selected models", () => {
    expect(
      buildFallbackEntries(
        [
          {
            id: "1",
            primaryModel: "fast",
            fallbackModels: Array.from({ length: 11 }, (_, i) => `target-${i}`),
          },
        ],
        models,
        [],
      ).error,
    ).toContain("at most 10");
  });
});
