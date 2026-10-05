import { describe, expect, it } from "vitest";

import { NOTE_QUERY } from "./__fixtures__/notes";
import { parseQuery, valueMatcher } from "./language";

describe("parseQuery", () => {
  it("splits clauses on whitespace, keeps quoted stretches, and only promotes known keys to fields", () => {
    expect(parseQuery(NOTE_QUERY, 'refund -Tag:"a b" foo:bar tag:')).toEqual([
      { kind: "text", value: "refund", from: 0, to: 6 },
      { kind: "field", field: "tag", negated: true, keyTo: 11, value: "a b", from: 7, to: 17 },
      { kind: "text", value: "foo:bar", from: 18, to: 25 },
      { kind: "field", field: "tag", negated: false, keyTo: 29, value: "", from: 26, to: 30 },
    ]);
  });
});

describe("valueMatcher", () => {
  it("requires a whole-value match, ignoring case, unless the pattern has wildcards", () => {
    expect(valueMatcher("billing")("BILLING")).toBe(true);
    expect(valueMatcher("bill")("billing")).toBe(false);
    expect(valueMatcher("*bill*")("research-billing-agent")).toBe(true);
    expect(valueMatcher("bill*")("billing")).toBe(true);
    expect(valueMatcher("*age")("triage")).toBe(true);
    expect(valueMatcher("res.arch")("research")).toBe(false);
  });

  it("matches glob segments in order without overlapping the exact anchors", () => {
    expect(valueMatcher("a*a")("aa")).toBe(true);
    expect(valueMatcher("a*a")("a")).toBe(false);
    expect(valueMatcher("**")("x")).toBe(valueMatcher("*")("x"));
    expect(valueMatcher("a*b*c")("a-c-b-c")).toBe(true);
    expect(valueMatcher("a*c*b")("a-c-b-c")).toBe(false);
  });

  it("matches adversarial globs without catastrophic backtracking", () => {
    const startedAt = performance.now();
    const matched = valueMatcher(`${"a*".repeat(30)}b`)("a".repeat(5000));
    expect(matched).toBe(false);
    expect(performance.now() - startedAt).toBeLessThan(100);
  });
});
