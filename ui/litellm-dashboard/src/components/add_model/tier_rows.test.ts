import { describe, expect, it } from "vitest";

import type { ActiveTierRow, CustomTierSet, TierRow } from "./tier_rows";
import {
  CUSTOM_TIER_OMITTED_KEYS,
  CUSTOM_TIER_RESTRICTIONS,
  MAX_TIER_COUNT,
  activeTierName,
  activeTierRows,
  isBuiltInTierName,
  resolveComplexityDefaultModel,
  sameTierIdentity,
  tierRowById,
  getCustomTierRowsError,
  restoredBuiltInRows,
  tierParamsByRowId,
  tierRowByName,
} from "./tier_rows";

const tiers = { SIMPLE: ["a"], MEDIUM: ["b"], COMPLEX: ["c"], REASONING: ["d"] };

describe("activeTierRows", () => {
  it("reads the tier set as rows whose id is the canonical tier key, in severity order", () => {
    expect(activeTierRows({ tiers })).toEqual([
      { id: "SIMPLE", name: "SIMPLE", definition: "", models: ["a"], params: {} },
      { id: "MEDIUM", name: "MEDIUM", definition: "", models: ["b"], params: {} },
      { id: "COMPLEX", name: "COMPLEX", definition: "", models: ["c"], params: {} },
      { id: "REASONING", name: "REASONING", definition: "", models: ["d"], params: {} },
    ]);
  });

  it("gives a tier with no models an empty pool rather than dropping the row", () => {
    const withEmptyComplex = { tiers: { ...tiers, COMPLEX: [] } };
    const emptyComplexRow: ActiveTierRow = { id: "COMPLEX", name: "COMPLEX", definition: "", models: [], params: {} };
    expect(activeTierRows(withEmptyComplex)[2]).toEqual(emptyComplexRow);
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
    const padded: TierRow = { id: "1", name: "  AUDIT  ", definition: "", models: [] };
    expect(activeTierName(padded)).toBe("AUDIT");
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

const definedRow = (name: string, models: string[] = ["m"], definition = "what belongs here"): TierRow => ({
  id: name.toLowerCase(),
  name,
  definition,
  models,
});

const set = (rows: TierRow[], fallback?: string): CustomTierSet => ({
  tiers: rows,
  fallback_tier_id: fallback ?? rows[0]?.id ?? "",
});

describe("activeTierRows with an edited set", () => {
  it("reads the edited rows instead of the built-in record once a set is present", () => {
    const custom = set([definedRow("CASUAL"), definedRow("AUDIT")]);
    expect(activeTierRows({ tiers, custom_tier_set: custom }).map((r) => r.name)).toEqual(["CASUAL", "AUDIT"]);
  });

  it("prefers the fallback tier's pool for the default model, mirroring init_complexity_router_deployment", () => {
    const custom = set([definedRow("CASUAL", ["casual-model"]), definedRow("AUDIT", ["audit-model"])], "audit");
    expect(resolveComplexityDefaultModel({ tiers, custom_tier_set: custom })).toBe("audit-model");
  });
});

describe("CUSTOM_TIER_RESTRICTIONS", () => {
  it("gives every restriction a reason, since each one replaces or explains a control", () => {
    const reasons = Object.values(CUSTOM_TIER_RESTRICTIONS).map((restriction) => restriction.reason);
    expect(reasons.every((reason) => reason.length > 0)).toBe(true);
    expect(new Set(reasons).size).toBe(reasons.length);
  });

  it("collects every omitted key exactly once, so no key is dropped by two owners", () => {
    expect(new Set(CUSTOM_TIER_OMITTED_KEYS).size).toBe(CUSTOM_TIER_OMITTED_KEYS.length);
  });

  it("omits the keys the backend rejects beside tier_definitions", () => {
    expect(CUSTOM_TIER_OMITTED_KEYS).toEqual(
      expect.arrayContaining(["tier_labels", "escalation_keywords", "adaptive", "classifier_fallback"]),
    );
  });
});

describe("restoredBuiltInRows", () => {
  it("brings missing built-ins back in canonical order and leaves custom rows after them", () => {
    const restored = restoredBuiltInRows([definedRow("AUDIT"), { ...definedRow("COMPLEX"), id: "COMPLEX" }], tiers);
    expect(restored.map((r) => r.id)).toEqual(["SIMPLE", "MEDIUM", "COMPLEX", "REASONING", "audit"]);
  });

  it("keeps the models an already-present built-in row carries rather than the stale record", () => {
    const edited = { ...definedRow("SIMPLE", ["edited"]), id: "SIMPLE" };
    expect(restoredBuiltInRows([edited], tiers).find((r) => r.id === "SIMPLE")?.models).toEqual(["edited"]);
  });
});

describe("restoredBuiltInRows name collisions", () => {
  it("does not restore a built-in slot whose name a custom row already answers to", () => {
    const custom: TierRow[] = [
      { id: "uuid-1", name: "SIMPLE", definition: "operator took this name", models: ["a"] },
      { id: "MEDIUM", name: "MEDIUM", definition: "", models: ["b"] },
    ];
    const restored = restoredBuiltInRows(custom, tiers);
    const folded = restored.map((row) => row.name.toLowerCase());
    expect(new Set(folded).size).toBe(folded.length);
    expect(restored.filter((row) => row.name === "SIMPLE")).toHaveLength(1);
  });

  it("still restores the built-in slots nothing else claims", () => {
    const custom: TierRow[] = [{ id: "uuid-1", name: "AUDIT", definition: "d", models: ["a"] }];
    expect(restoredBuiltInRows(custom, tiers).map((row) => row.id)).toEqual([
      "SIMPLE",
      "MEDIUM",
      "COMPLEX",
      "REASONING",
      "uuid-1",
    ]);
  });
});

describe("getCustomTierRowsError", () => {
  it("accepts a complete set", () => {
    expect(getCustomTierRowsError(set([definedRow("CASUAL"), definedRow("AUDIT")]))).toBeNull();
  });

  it.each([
    [set([definedRow("CASUAL")]), "A tier set needs 2 to 8 tiers"],
    [
      set(Array.from({ length: MAX_TIER_COUNT + 1 }, (_, index) => definedRow(`T${index}`))),
      "A tier set needs 2 to 8 tiers",
    ],
    [set([definedRow(""), definedRow("AUDIT")], "audit"), "Name every tier"],
    [set([definedRow("AUDIT"), { ...definedRow("audit"), id: "second" }]), "Tier names must be unique, ignoring case"],
    [
      set([definedRow("CASUAL"), { ...definedRow("AUDIT"), definition: "  " }]),
      "Every custom tier needs a definition: it is the rubric the classifier routes on",
    ],
    [set([definedRow("CASUAL"), definedRow("AUDIT")], "gone"), "Pick a Fallback Tier for classifier failures"],
  ])("reports the row problem the backend would reject", (customTierSet, expected) => {
    expect(getCustomTierRowsError(customTierSet)).toBe(expected);
  });

  it("lets a built-in name inherit its definition, which is the one blank the backend allows", () => {
    expect(getCustomTierRowsError(set([{ ...definedRow("SIMPLE"), definition: "" }, definedRow("AUDIT")]))).toBeNull();
  });
});

describe("tierParamsByRowId", () => {
  const rows = [
    { id: "SIMPLE", name: "SIMPLE", definition: "", models: ["a"] },
    { id: "stored-1", name: "SECURITY_REVIEW", definition: "audits", models: ["b"] },
  ];

  it("re-keys a stored tier name onto the ephemeral row id the editor reads", () => {
    const stored = { SECURITY_REVIEW: { b: { reasoning_effort: "high" } } };
    expect(tierParamsByRowId(stored, rows)).toEqual({ "stored-1": { b: { reasoning_effort: "high" } } });
  });

  it("leaves a built-in tier untouched, because its row id is already the tier name", () => {
    const stored = { SIMPLE: { a: { reasoning_effort: "low" } } };
    expect(tierParamsByRowId(stored, rows)).toEqual(stored);
  });

  it("passes a tier this editor does not render straight through instead of dropping its params", () => {
    const stored = { DEEP_RESEARCH: { c: { reasoning_effort: "high" } } };
    expect(tierParamsByRowId(stored, rows)).toEqual(stored);
  });

  it("returns nothing when there are no stored params, keeping the key out of the payload", () => {
    expect(tierParamsByRowId(undefined, rows)).toBeUndefined();
  });
});
