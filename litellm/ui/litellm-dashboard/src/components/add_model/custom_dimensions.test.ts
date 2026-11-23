import { describe, expect, it } from "vitest";
import { customDimensionsError, hydrateCustomDimensions, serializeCustomDimensions } from "./custom_dimensions";

describe("custom dimension drafts", () => {
  it.each([
    [[{ name: "domain", weight: 0.2, keywords: [" orbitmesh "] }]],
    [[{ name: "domain", weight: 0.2, patterns: ["abc"], keywords: [], scoring_mode: "binary" }]],
    [[{ name: "domain", weight: 0.2, keywords: ["a", "b"], scoring_mode: "match_count" }]],
    [[]],
  ])("round-trips stored optional fields and matcher text without normalization: %j", (raw) => {
    const rows = hydrateCustomDimensions(raw);
    expect(rows).toBeDefined();
    expect(serializeCustomDimensions(rows!)).toEqual(raw);
  });

  it("does not insert a default mode or custom list on load", () => {
    expect(hydrateCustomDimensions(undefined)).toBeUndefined();
    const rows = hydrateCustomDimensions([{ name: "domain", weight: 0.2, keywords: ["a"] }])!;
    expect(rows[0].scoring_mode).toBeUndefined();
    expect(customDimensionsError(rows)).toBeNull();
  });

  it.each([
    { name: "", weight: 0.2, keywords: ["a"] },
    { name: "codePresence", weight: 0.2, keywords: ["a"] },
    { name: "bad name", weight: 0.2, keywords: ["a"] },
    { name: "domain", weight: 0, keywords: ["a"] },
    { name: "domain", weight: 0.2, keywords: [] },
    { name: "domain", weight: 0.2, keywords: [" "] },
    { name: "domain", weight: 0.2, keywords: ["a".repeat(257)] },
  ])("rejects an invalid draft %j", (row) => {
    expect(customDimensionsError([{ ...row, id: "draft" }])).not.toBeNull();
  });

  it("rejects duplicate names and aggregate matcher limits", () => {
    const row = { id: "a", name: "domain", weight: 0.2, keywords: ["a"] };
    expect(customDimensionsError([row, { ...row, id: "b", name: "DOMAIN" }])).not.toBeNull();
    expect(customDimensionsError([{ ...row, keywords: Array(33).fill("a") }])).not.toBeNull();
    expect(customDimensionsError([{ ...row, keywords: Array(32).fill("a".repeat(256)) }])).not.toBeNull();
  });
});
