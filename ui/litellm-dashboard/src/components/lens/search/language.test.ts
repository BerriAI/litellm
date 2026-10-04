import { describe, expect, it } from "vitest";

import { note, NOTE_QUERY, notes } from "./__fixtures__/notes";
import { fieldValues, filterItems, parseQuery } from "./language";

const ids = (query: string) => filterItems(NOTE_QUERY, notes, query).map((n) => n.id);

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

describe("filterItems", () => {
  it("returns every item for an empty or blank query", () => {
    expect(filterItems(NOTE_QUERY, notes, "   ")).toBe(notes);
  });

  it("matches free text against the language's free-text fields, ignoring case, with every term required", () => {
    expect(ids("REFUND")).toEqual(["aaa111"]);
    expect(ids("lead")).toEqual(["bbb222"]);
    expect(ids("ccc3")).toEqual(["ccc333"]);
    expect(ids("vector research")).toEqual(["bbb222"]);
    expect(ids("vector refund")).toEqual([]);
    expect(ids("billing")).toEqual([]);
  });

  it("keeps a quoted phrase together", () => {
    expect(ids('"my refund"')).toEqual(["aaa111"]);
    expect(ids('"refund my"')).toEqual([]);
  });

  it("matches a multi-valued field when any value matches, and negation when none does", () => {
    expect(ids("tag:triage")).toEqual(["aaa111"]);
    expect(ids("-tag:triage")).toEqual(["bbb222", "ccc333"]);
    expect(ids("TAG:Cron")).toEqual(["ccc333"]);
  });

  it("requires a whole-value match unless the value has wildcards", () => {
    expect(ids("tag:bill")).toEqual([]);
    expect(ids("tag:*bill*")).toEqual(["aaa111"]);
    expect(ids("tag:bill*")).toEqual(["aaa111"]);
    expect(ids("tag:*age")).toEqual(["aaa111"]);
    expect(ids("-tag:*search*")).toEqual(["aaa111", "ccc333"]);
    expect(ids("title:res.arch_lead")).toEqual([]);
  });

  it("matches glob segments case-insensitively without overlapping exact anchors", () => {
    const billing = note({ id: "billing", tags: ["BILLING-agent"] });
    const research = note({ id: "research", tags: ["research-billing-agent"] });
    const doubled = note({ id: "doubled", tags: ["agent"] });
    const globNotes = [billing, research, doubled];

    expect(filterItems(NOTE_QUERY, globNotes, "tag:*bill*")).toEqual([billing, research]);
    expect(filterItems(NOTE_QUERY, globNotes, "tag:billing*")).toEqual([billing]);
    expect(filterItems(NOTE_QUERY, globNotes, "tag:*agent")).toEqual(globNotes);
    expect(filterItems(NOTE_QUERY, [note({ title: "aa" })], "title:a*a")).toHaveLength(1);
    expect(filterItems(NOTE_QUERY, [note({ title: "a" })], "title:a*a")).toHaveLength(0);
    expect(filterItems(NOTE_QUERY, globNotes, "tag:**")).toEqual(filterItems(NOTE_QUERY, globNotes, "tag:*"));
  });

  it("matches adversarial globs without catastrophic backtracking", () => {
    const longValue = note({ tags: ["a".repeat(5000)] });
    const pattern = `tag:${"a*".repeat(30)}b`;
    const startedAt = performance.now();
    const matches = filterItems(NOTE_QUERY, [longValue], pattern);
    const elapsed = performance.now() - startedAt;

    expect(matches).toEqual([]);
    expect(elapsed).toBeLessThan(100);
  });

  it("searches quoted values with spaces in a free-form field", () => {
    expect(ids('body:"*vector stores"')).toEqual(["bbb222"]);
    expect(ids("id:bbb222")).toEqual(["bbb222"]);
  });

  it("combines field clauses with free text", () => {
    expect(ids("tag:*e* -tag:researcher refund")).toEqual(["aaa111"]);
  });

  it("ignores a key still waiting for its value and treats unknown keys as text", () => {
    expect(ids("tag:")).toEqual(["aaa111", "bbb222", "ccc333"]);
    expect(ids("refund tag:")).toEqual(["aaa111"]);
    expect(ids("foo:bar")).toEqual([]);
    expect(filterItems(NOTE_QUERY, [note({ title: "foo:bar" })], "foo:bar")).toHaveLength(1);
  });
});

describe("fieldValues", () => {
  it("lists each distinct non-empty value once, sorted", () => {
    expect(fieldValues(NOTE_QUERY, notes, "tag")).toEqual(["billing", "cron", "researcher", "triage"]);
    expect(fieldValues(NOTE_QUERY, [...notes, note({ title: "" })], "title")).toEqual([
      "health",
      "research_lead",
      "support",
    ]);
    expect(fieldValues(NOTE_QUERY, [note({ tags: [] })], "tag")).toEqual([]);
  });
});
