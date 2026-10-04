import { describe, expect, it } from "vitest";

import { note, type NoteField, NOTE_QUERY, notes } from "./__fixtures__/notes";
import { suggest, type Suggestion, type SuggestionMenu } from "./suggestions";

const atEnd = (text: string, source = notes) => suggest(NOTE_QUERY, text, text.length, source);
const items = (menu: SuggestionMenu<NoteField> | null): Suggestion<NoteField>[] =>
  menu?.groups.flatMap((g) => g.items) ?? [];
const labels = (menu: SuggestionMenu<NoteField> | null) => items(menu).map((item) => item.label);
const apply = (text: string, item: Suggestion<NoteField>) =>
  text.slice(0, item.from) + item.insert + text.slice(item.to);

describe("suggest", () => {
  it("offers every field, grouped in declaration order, in an empty box", () => {
    const menu = atEnd("");
    expect(menu?.groups.map((g) => g.heading)).toEqual(["Note", "Content", "Identity"]);
    expect(labels(menu)).toEqual(["title", "tag", "body", "id"]);
    expect(menu?.showOperators).toBe(false);
  });

  it("narrows fields by the typed prefix and completes the key with a colon", () => {
    const menu = atEnd("refund TA");
    expect(labels(menu)).toEqual(["tag"]);
    expect(apply("refund TA", items(menu)[0])).toBe("refund tag:");
    expect(items(menu)[0].completesClause).toBe(false);
  });

  it("keeps a leading minus when completing a negated key", () => {
    expect(apply("-ta", items(atEnd("-ta"))[0])).toBe("-tag:");
  });

  it("offers nothing for free text that names no field", () => {
    expect(atEnd("refund")).toBeNull();
    expect(atEnd("foo:bar")).toBeNull();
  });

  it("offers the loaded values and the operator help once a key has its colon", () => {
    const menu = atEnd("tag:");
    expect(menu?.groups.map((g) => g.heading)).toEqual(["tag"]);
    expect(labels(menu)).toEqual(["billing", "cron", "researcher", "triage"]);
    expect(items(menu).every((item) => item.field === "tag")).toBe(true);
    expect(menu?.showOperators).toBe(true);
  });

  it("narrows values by substring, ignoring wildcards, and drops operator help once a value is typed", () => {
    const menu = atEnd("-tag:*AR*");
    expect(labels(menu)).toEqual(["researcher"]);
    expect(menu?.showOperators).toBe(false);
    expect(atEnd("tag:zzz")).toBeNull();
  });

  it("replaces only the value, adds a separating space at the end, and finishes the clause", () => {
    const item = items(atEnd("-tag:bil"))[0];
    expect(apply("-tag:bil", item)).toBe("-tag:billing ");
    expect(item.completesClause).toBe(true);
    const text = "tag:tri refund";
    expect(apply(text, items(suggest(NOTE_QUERY, text, 7, notes))[0])).toBe("tag:triage refund");
  });

  it("quotes a value that contains spaces", () => {
    const spaced = [note({ title: "Order lookup" })];
    expect(apply("title:", items(atEnd("title:", spaced))[0])).toBe('title:"Order lookup" ');
  });

  it("does not list values for free-form fields but still explains the operators", () => {
    const menu = atEnd("body:");
    expect(menu?.groups).toEqual([]);
    expect(menu?.showOperators).toBe(true);
    expect(atEnd("body:x")).toBeNull();
  });

  it("offers fields in the gap between tokens, but nothing mid-token", () => {
    const text = "refund  tag:cron";
    expect(apply(text, items(suggest(NOTE_QUERY, text, 7, notes))[1])).toBe("refund tag: tag:cron");
    expect(suggest(NOTE_QUERY, text, 3, notes)).toBeNull();
    expect(suggest(NOTE_QUERY, "tag:cron", 0, notes)).toBeNull();
  });
});
