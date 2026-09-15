import { describe, expect, it } from "vitest";
import { mergeCustomEntities, normalizeCustomEntityName } from "./piiCustomEntity";

describe("normalizeCustomEntityName", () => {
  it.each([
    ["se personnummer", "SE_PERSONNUMMER"],
    ["  no-fodselsnummer ", "NO_FODSELSNUMMER"],
  ])("normalizes %s", (raw, expected) => {
    expect(normalizeCustomEntityName(raw)).toBe(expected);
  });

  it.each(["", "bad!name"])("rejects %s", (raw) => {
    expect(normalizeCustomEntityName(raw)).toBeNull();
  });
});

describe("mergeCustomEntities", () => {
  it("keeps supported order, dedupes, and appends unknown selected entities", () => {
    expect(mergeCustomEntities(["PERSON", "EMAIL"], ["EMAIL", "NO_FODSELSNUMMER", "PERSON", "SE_PERSONNUMMER"])).toEqual([
      "PERSON",
      "EMAIL",
      "NO_FODSELSNUMMER",
      "SE_PERSONNUMMER",
    ]);
  });
});
