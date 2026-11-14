import { describe, expect, it } from "vitest";

import { hydrateKeywordTierRules, serializeKeywordTierRules } from "./complexity_router_keywords";

describe("hydrateKeywordTierRules", () => {
  it("keeps a rule whose tier is operator-defined instead of silently deleting it on edit", () => {
    const stored = [
      { keywords: ["invoice"], tier: "MEDIUM" },
      { keywords: ["pentest", "vulnerability"], tier: "SECURITY_REVIEW" },
    ];
    expect(hydrateKeywordTierRules(stored)).toEqual([
      { id: "stored-0", keywords: ["invoice"], tier: "MEDIUM" },
      { id: "stored-1", keywords: ["pentest", "vulnerability"], tier: "SECURITY_REVIEW" },
    ]);
  });

  it("round-trips through serialize without loss", () => {
    const stored = [{ keywords: ["pentest"], tier: "SECURITY_REVIEW" }];
    expect(serializeKeywordTierRules(hydrateKeywordTierRules(stored))).toEqual(stored);
  });

  it("still drops rows that are not rules at all", () => {
    expect(hydrateKeywordTierRules([{ keywords: [], tier: "MEDIUM" }, { keywords: ["x"] }, "junk", null])).toEqual([]);
  });
});
