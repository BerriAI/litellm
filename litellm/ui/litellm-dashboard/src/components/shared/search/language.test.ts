import { describe, expect, it } from "vitest";

import { EXACT_NOTE_QUERY, NEGATION_NOTE_QUERY, NOTE_QUERY, WILDCARD_NOTE_QUERY } from "./__fixtures__/notes";
import { exactMatcher, languageOps, parseQuery, valueMatcher } from "./language";

describe("parseQuery", () => {
  it("splits clauses on whitespace, keeps quoted stretches, and only promotes known keys to fields", () => {
    expect(parseQuery(NOTE_QUERY, 'refund -Tag:"a b" foo:bar tag:')).toEqual([
      { kind: "text", value: "refund", from: 0, to: 6 },
      { kind: "field", field: "tag", op: "neq", keyTo: 11, value: "a b", from: 7, to: 17 },
      { kind: "text", value: "foo:bar", from: 18, to: 25 },
      { kind: "field", field: "tag", op: "eq", keyTo: 29, value: "", from: 26, to: 30 },
    ]);
  });

  it("picks the op from the dash and the wildcard when the language honors them", () => {
    const ops = (text: string) => parseQuery(NOTE_QUERY, text).map((c) => (c.kind === "field" ? c.op : c.kind));
    expect(ops("tag:a -tag:b tag:*c -tag:d*")).toEqual(["eq", "neq", "glob", "nglob"]);
  });

  it("keeps a dash as text and a star as a literal for an equality-only language", () => {
    expect(parseQuery(EXACT_NOTE_QUERY, "-tag:a tag:b*")).toEqual([
      { kind: "text", value: "-tag:a", from: 0, to: 6 },
      { kind: "field", field: "tag", op: "eq", keyTo: 10, value: "b*", from: 7, to: 13 },
    ]);
  });

  it.each([
    [NOTE_QUERY, ["nglob", "neq", "glob", "eq"]],
    [EXACT_NOTE_QUERY, ["text", "text", "eq", "eq"]],
    [NEGATION_NOTE_QUERY, ["neq", "neq", "eq", "eq"]],
    [WILDCARD_NOTE_QUERY, ["text", "text", "glob", "eq"]],
  ])("reads each operator form the way the language's capabilities allow (%#)", (language, expected) => {
    const ops = parseQuery(language, "-tag:research* -tag:research tag:research* tag:research").map((c) =>
      c.kind === "field" ? c.op : c.kind,
    );
    expect(ops).toEqual(expected);
  });

  it("keeps the star in the value and the dash in the text whatever the capabilities", () => {
    expect(parseQuery(NEGATION_NOTE_QUERY, "-Tag:research*")).toEqual([
      { kind: "field", field: "tag", op: "neq", keyTo: 4, value: "research*", from: 0, to: 14 },
    ]);
    expect(parseQuery(WILDCARD_NOTE_QUERY, '-tag:"a b" tag:"a *"')).toEqual([
      { kind: "text", value: '-tag:"a b"', from: 0, to: 10 },
      { kind: "field", field: "tag", op: "glob", keyTo: 14, value: "a *", from: 11, to: 20 },
    ]);
  });
});

describe("languageOps", () => {
  it("lists only the ops a language can express", () => {
    expect(languageOps(NOTE_QUERY)).toEqual(["eq", "neq", "glob", "nglob"]);
    expect(languageOps(EXACT_NOTE_QUERY)).toEqual(["eq"]);
    expect(languageOps({ ...NOTE_QUERY, ops: { negation: true, wildcard: false } })).toEqual(["eq", "neq"]);
    expect(languageOps({ ...NOTE_QUERY, ops: { negation: false, wildcard: true } })).toEqual(["eq", "glob"]);
  });
});

describe("exactMatcher", () => {
  it("compares the whole value ignoring case and treats a star as a character", () => {
    expect(exactMatcher("A*b")("a*B")).toBe(true);
    expect(exactMatcher("a*b")("axb")).toBe(false);
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
