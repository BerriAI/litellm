import { Braces, Hash, Tag, Type } from "lucide-react";

import type { FieldSpec, QueryLanguage } from "../language";

export interface Note {
  readonly id: string;
  readonly title: string;
  readonly tags: readonly string[];
  readonly body: string;
}

const NOTE_FIELDS = {
  title: { group: "Note", icon: Type, read: (note) => [note.title], suggestValues: true },
  tag: { group: "Note", icon: Tag, read: (note) => note.tags, suggestValues: true },
  body: { group: "Content", icon: Braces, read: (note) => [note.body], suggestValues: false },
  id: { group: "Identity", icon: Hash, read: (note) => [note.id], suggestValues: false },
} as const satisfies Record<string, FieldSpec<Note>>;

export type NoteField = keyof typeof NOTE_FIELDS;

export const NOTE_QUERY: QueryLanguage<Note, NoteField> = {
  fields: NOTE_FIELDS,
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
