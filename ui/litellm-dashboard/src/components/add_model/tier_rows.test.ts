import { describe, expect, it } from "vitest";

import {
  activeTierName,
  activeTierRows,
  isBuiltInTierName,
  resolveComplexityDefaultModel,
  sameTierIdentity,
  tierRowById,
  tierRowByName,
} from "./tier_rows";

const tiers = { SIMPLE: ["a"], MEDIUM: ["b"], COMPLEX: ["c"], REASONING: ["d"] };

describe("activeTierRows", () => {
  it("reads the tier set as rows whose id is the canonical tier key, in severity order", () => {
    expect(activeTierRows({ tiers })).toEqual([
      { id: "SIMPLE", name: "SIMPLE", models: ["a"] },
      { id: "MEDIUM", name: "MEDIUM", models: ["b"] },
      { id: "COMPLEX", name: "COMPLEX", models: ["c"] },
      { id: "REASONING", name: "REASONING", models: ["d"] },
    ]);
  });

  it("gives a tier with no models an empty pool rather than dropping the row", () => {
    expect(activeTierRows({ tiers: { ...tiers, COMPLEX: [] } })[2]).toEqual({
      id: "COMPLEX",
      name: "COMPLEX",
      models: [],
    });
  });

  it("finds a row by id and by name", () => {
    const rows = activeTierRows({ tiers });
    expect(tierRowById(rows, "MEDIUM")?.models).toEqual(["b"]);
    expect(tierRowById(rows, undefined)).toBeUndefined();
    expect(tierRowByName(rows, " medium ")?.id).toBe("MEDIUM");
  });
});

describe("sameTierIdentity", () => {
  it.each([
    ["AUDIT", "audit", true],
    ["AUDIT", " audit ", true],
    ["AUDIT", "AUDITS", false],
  ])("compares %s and %s casefold, matching the backend's uniqueness rule", (left, right, expected) => {
    expect(sameTierIdentity(left, right)).toBe(expected);
  });

  it("recognises the four built-in names regardless of case", () => {
    expect(["SIMPLE", "medium", "Complex", "REASONING"].every(isBuiltInTierName)).toBe(true);
    expect(isBuiltInTierName("SECURITY_REVIEW")).toBe(false);
  });

  it("trims a row name, since the backend matches fallback_tier and keyword rules exactly", () => {
    expect(activeTierName({ id: "1", name: "  AUDIT  ", models: [] })).toBe("AUDIT");
  });
});

describe("resolveComplexityDefaultModel", () => {
  it("mirrors init_complexity_router_deployment: a pin wins, then MEDIUM, then SIMPLE", () => {
    expect(resolveComplexityDefaultModel({ tiers }, "pinned")).toBe("pinned");
    expect(resolveComplexityDefaultModel({ tiers })).toBe("b");
    expect(resolveComplexityDefaultModel({ tiers: { ...tiers, MEDIUM: [] } })).toBe("a");
  });

  it("resolves to nothing rather than falling through to COMPLEX, which the backend never picks", () => {
    expect(resolveComplexityDefaultModel({ tiers: { ...tiers, SIMPLE: [], MEDIUM: [] } })).toBeUndefined();
  });
});
