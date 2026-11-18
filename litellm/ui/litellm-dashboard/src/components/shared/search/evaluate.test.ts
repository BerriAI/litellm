import { describe, expect, it } from "vitest";

import {
  EXACT_NOTE_QUERY,
  NEGATION_NOTE_QUERY,
  note,
  NOTE_INDEX,
  NOTE_QUERY,
  notes,
  WILDCARD_NOTE_QUERY,
} from "./__fixtures__/notes";
import { evaluate, fieldValues, filterItems } from "./evaluate";

const ids = (text: string) => filterItems(NOTE_QUERY, NOTE_INDEX, notes, text).map((n) => n.id);

describe("evaluate", () => {
  it("returns the same array for an empty query", () => {
    expect(evaluate(NOTE_INDEX, notes, { text: [], filters: [] })).toBe(notes);
  });

  it("applies each op the way the wire contract defines it", () => {
    const by = (op: "eq" | "neq" | "glob" | "nglob", value: string) =>
      evaluate(NOTE_INDEX, notes, { text: [], filters: [{ field: "tag", op, value }] }).map((n) => n.id);
    expect(by("eq", "TRIAGE")).toEqual(["aaa111"]);
    expect(by("eq", "tri")).toEqual([]);
    expect(by("neq", "triage")).toEqual(["bbb222", "ccc333"]);
    expect(by("glob", "*search*")).toEqual(["bbb222"]);
    expect(by("nglob", "*search*")).toEqual(["aaa111", "ccc333"]);
  });

  it("treats a star as a literal under eq and neq, so an equality-only query never globs", () => {
    const starred = [note({ id: "star", title: "a*b" }), note({ id: "x", title: "axb" })];
    const by = (op: "eq" | "neq") =>
      evaluate(NOTE_INDEX, starred, { text: [], filters: [{ field: "title", op, value: "a*b" }] }).map((n) => n.id);
    expect(by("eq")).toEqual(["star"]);
    expect(by("neq")).toEqual(["x"]);
    expect(filterItems(EXACT_NOTE_QUERY, NOTE_INDEX, starred, "title:a*b").map((n) => n.id)).toEqual(["star"]);
  });

  it("follows the language's capabilities from the typed text to the matched items", () => {
    const tagged = [...notes, note({ id: "star", tags: ["research*"] })];
    const hits = (language: typeof NOTE_QUERY, text: string) =>
      filterItems(language, NOTE_INDEX, tagged, text).map((n) => n.id);
    expect(hits(NOTE_QUERY, "-tag:research*")).toEqual(["aaa111", "ccc333"]);
    expect(hits(EXACT_NOTE_QUERY, "tag:research*")).toEqual(["star"]);
    expect(hits(EXACT_NOTE_QUERY, "-tag:researcher")).toEqual([]);
    expect(hits(NEGATION_NOTE_QUERY, "-tag:researcher")).toEqual(["aaa111", "ccc333", "star"]);
    expect(hits(NEGATION_NOTE_QUERY, "-tag:research*")).toEqual(["aaa111", "bbb222", "ccc333"]);
    expect(hits(WILDCARD_NOTE_QUERY, "tag:research*")).toEqual(["bbb222", "star"]);
    expect(hits(WILDCARD_NOTE_QUERY, "-tag:researcher")).toEqual([]);
  });

  it("requires every term and every filter", () => {
    expect(
      evaluate(NOTE_INDEX, notes, { text: ["refund"], filters: [{ field: "tag", op: "eq", value: "billing" }] }),
    ).toHaveLength(1);
    expect(
      evaluate(NOTE_INDEX, notes, { text: ["refund"], filters: [{ field: "tag", op: "eq", value: "cron" }] }),
    ).toHaveLength(0);
    expect(evaluate(NOTE_INDEX, notes, { text: ["vector", "refund"], filters: [] })).toHaveLength(0);
  });
});

describe("filterItems", () => {
  it("matches free text against the index's free-text fields only, ignoring case", () => {
    expect(ids("REFUND")).toEqual(["aaa111"]);
    expect(ids("ccc3")).toEqual(["ccc333"]);
    expect(ids("billing")).toEqual([]);
    expect(ids('"my refund"')).toEqual(["aaa111"]);
    expect(ids('"refund my"')).toEqual([]);
  });

  it("matches a multi-valued field when any value matches, and negation when none does", () => {
    expect(ids("tag:triage")).toEqual(["aaa111"]);
    expect(ids("-tag:triage")).toEqual(["bbb222", "ccc333"]);
    expect(ids("-tag:*search*")).toEqual(["aaa111", "ccc333"]);
  });

  it("searches quoted values with spaces in a free-form field and combines with free text", () => {
    expect(ids('body:"*vector stores"')).toEqual(["bbb222"]);
    expect(ids("tag:*e* -tag:researcher refund")).toEqual(["aaa111"]);
  });

  it("ignores a key still waiting for its value and treats unknown keys as text", () => {
    expect(ids("tag:")).toEqual(["aaa111", "bbb222", "ccc333"]);
    expect(ids("refund tag:")).toEqual(["aaa111"]);
    expect(ids("foo:bar")).toEqual([]);
    expect(filterItems(NOTE_QUERY, NOTE_INDEX, [note({ title: "foo:bar" })], "foo:bar")).toHaveLength(1);
  });
});

describe("fieldValues", () => {
  it("lists each distinct non-empty value once, sorted", () => {
    expect(fieldValues(NOTE_INDEX, notes, "tag")).toEqual(["billing", "cron", "researcher", "triage"]);
    expect(fieldValues(NOTE_INDEX, [...notes, note({ title: "" })], "title")).toEqual([
      "health",
      "research_lead",
      "support",
    ]);
    expect(fieldValues(NOTE_INDEX, [note({ tags: [] })], "tag")).toEqual([]);
  });
});
