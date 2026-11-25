import { describe, expect, it } from "vitest";

import {
  EXACT_NOTE_QUERY,
  NEGATION_NOTE_QUERY,
  type Note,
  note,
  NOTE_INDEX,
  type NoteField,
  NOTE_QUERY,
  notes,
  WILDCARD_NOTE_QUERY,
} from "./__fixtures__/notes";
import { fieldValues } from "./evaluate";
import { completingField, completingPrefix, suggest, type Suggestion, type SuggestionMenu } from "./suggestions";
import type { FieldValues } from "./valueSource";

const lookupIn =
  (source: readonly Note[]) =>
  (field: NoteField): FieldValues => ({ values: fieldValues(NOTE_INDEX, source, field), loading: false });
const at = (text: string, cursor: number, source = notes) => suggest(NOTE_QUERY, text, cursor, lookupIn(source));
const atEnd = (text: string, source = notes) => at(text, text.length, source);
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

  it("offers the source's values and the operator help once a key has its colon", () => {
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

  it("keeps the menu open while values load, and closes it once an empty answer arrives", () => {
    const loading = suggest(NOTE_QUERY, "tag:zzz", 7, () => ({ values: [], loading: true }));
    expect(loading).toEqual({ groups: [], showOperators: false, loading: true });
    expect(suggest(NOTE_QUERY, "tag:zzz", 7, () => ({ values: [], loading: false }))).toBeNull();
  });

  it("replaces only the value, adds a separating space at the end, and finishes the clause", () => {
    const item = items(atEnd("-tag:bil"))[0];
    expect(apply("-tag:bil", item)).toBe("-tag:billing ");
    expect(item.completesClause).toBe(true);
    const text = "tag:tri refund";
    expect(apply(text, items(at(text, 7))[0])).toBe("tag:triage refund");
  });

  it("quotes a value that contains spaces", () => {
    const spaced = [note({ title: "Order lookup" })];
    expect(apply("title:", items(atEnd("title:", spaced))[0])).toBe('title:"Order lookup" ');
  });

  it("never asks the source for a free-form field, but still explains the operators", () => {
    const asked: NoteField[] = [];
    const menu = suggest(NOTE_QUERY, "body:", 5, (field) => {
      asked.push(field);
      return { values: ["leak"], loading: false };
    });
    expect(asked).toEqual([]);
    expect(menu?.groups).toEqual([]);
    expect(menu?.showOperators).toBe(true);
    expect(atEnd("body:x")).toBeNull();
  });

  it("skips the operator help and the negated key for an equality-only language", () => {
    const exact = (text: string) => suggest(EXACT_NOTE_QUERY, text, text.length, lookupIn(notes));
    expect(exact("tag:")?.showOperators).toBe(false);
    expect(labels(exact("tag:"))).toEqual(["billing", "cron", "researcher", "triage"]);
    expect(exact("-ta")).toBeNull();
    expect(completingPrefix(EXACT_NOTE_QUERY, "tag:a*", 6)).toBe("a*");
    expect(completingPrefix(NOTE_QUERY, "tag:a*", 6)).toBe("a");
    expect(labels(exact("tag:*"))).toEqual([]);
  });

  it("keeps a literal star in the value prefix unless the language has wildcards", () => {
    const starred = [...notes, note({ tags: ["a*b", "ab"] })];
    const menu = (language: typeof NOTE_QUERY) => suggest(language, "tag:a*", 6, lookupIn(starred));
    expect(labels(menu(NEGATION_NOTE_QUERY))).toEqual(["a*b"]);
    expect(labels(menu(EXACT_NOTE_QUERY))).toEqual(["a*b"]);
    expect(labels(menu(WILDCARD_NOTE_QUERY))).toEqual(["a*b", "ab", "researcher", "triage"]);
    expect(completingPrefix(NEGATION_NOTE_QUERY, "tag:a*", 6)).toBe("a*");
    expect(completingPrefix(WILDCARD_NOTE_QUERY, "tag:a*", 6)).toBe("a");
  });

  it("offers a negated key only when the language can negate", () => {
    const fields = (language: typeof NOTE_QUERY) => suggest(language, "-ta", 3, lookupIn(notes));
    expect(apply("-ta", items(fields(NEGATION_NOTE_QUERY))[0])).toBe("-tag:");
    expect(fields(WILDCARD_NOTE_QUERY)).toBeNull();
    expect(suggest(WILDCARD_NOTE_QUERY, "-tag:", 5, lookupIn(notes))).toBeNull();
    expect(suggest(NEGATION_NOTE_QUERY, "tag:", 4, lookupIn(notes))?.showOperators).toBe(true);
    expect(suggest(WILDCARD_NOTE_QUERY, "tag:", 4, lookupIn(notes))?.showOperators).toBe(true);
  });

  it("offers fields in the gap between tokens, but nothing mid-token", () => {
    const text = "refund  tag:cron";
    expect(apply(text, items(at(text, 7))[1])).toBe("refund tag: tag:cron");
    expect(at(text, 3)).toBeNull();
    expect(at("tag:cron", 0)).toBeNull();
  });
});

describe("completingField", () => {
  it("names the enumerable field whose value sits at the cursor, with the typed prefix", () => {
    expect(completingField(NOTE_QUERY, "refund -tag:*bi", 15)).toBe("tag");
    expect(completingPrefix(NOTE_QUERY, "refund -tag:*bi", 15)).toBe("bi");
    expect(completingField(NOTE_QUERY, "tag:cron refund", 15)).toBeNull();
    expect(completingField(NOTE_QUERY, "tag:cron", 2)).toBeNull();
    expect(completingField(NOTE_QUERY, "body:x", 6)).toBeNull();
    expect(completingPrefix(NOTE_QUERY, "ta", 2)).toBe("");
  });
});
