import { Braces, Hash, Tag, Type } from "lucide-react";

import type { ClientIndex } from "../evaluate";
import { ALL_OPERATORS, EQUALITY_ONLY, type FieldSpec, type QueryLanguage } from "../language";

export interface Note {
  readonly id: string;
  readonly title: string;
  readonly tags: readonly string[];
  readonly body: string;
}

const NOTE_FIELDS = {
  title: { group: "Note", icon: Type, suggestValues: true },
  tag: { group: "Note", icon: Tag, suggestValues: true },
  body: { group: "Content", icon: Braces, suggestValues: false },
  id: { group: "Identity", icon: Hash, suggestValues: false },
} as const satisfies Record<string, FieldSpec>;

export type NoteField = keyof typeof NOTE_FIELDS;

export const NOTE_QUERY: QueryLanguage<NoteField> = { fields: NOTE_FIELDS, ops: ALL_OPERATORS };
/** The same vocabulary over a backend that only filters by equality. */
export const EXACT_NOTE_QUERY: QueryLanguage<NoteField> = { fields: NOTE_FIELDS, ops: EQUALITY_ONLY };
export const NEGATION_NOTE_QUERY: QueryLanguage<NoteField> = {
  fields: NOTE_FIELDS,
  ops: { negation: true, wildcard: false },
};
export const WILDCARD_NOTE_QUERY: QueryLanguage<NoteField> = {
  fields: NOTE_FIELDS,
  ops: { negation: false, wildcard: true },
};

export const NOTE_INDEX: ClientIndex<Note, NoteField> = {
  read: {
    title: (note) => [note.title],
    tag: (note) => note.tags,
    body: (note) => [note.body],
    id: (note) => [note.id],
  },
  freeText: (note) => [note.id, note.body, note.title],
};

export const note = (overrides: Partial<Note>): Note => ({
  id: "note",
  title: "note",
  tags: [],
  body: "",
  ...overrides,
});

const refundOverrides = { id: "aaa111", title: "support", body: "Where is my refund?", tags: ["billing", "triage"] };
const refund = note(refundOverrides);
const researchOverrides = { id: "bbb222", title: "research_lead", body: "Compare vector stores", tags: ["researcher"] };
const research = note(researchOverrides);
const plain = note({ id: "ccc333", title: "health", tags: ["cron"] });
export const notes = [refund, research, plain];
