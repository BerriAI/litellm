import type { DatasetCase, DatasetToolCall, SkipReason } from "./types";

export const SKIP_REASON_TEXT: Readonly<Record<SkipReason, string>> = {
  duplicate: "Already in the dataset",
  no_content: "No conversation or reply was recorded",
  too_large: "Too large to store as one case",
  over_limit: "The dataset is full",
};

export interface Draft {
  readonly excluded: ReadonlySet<string>;
  readonly expected: ReadonlyMap<string, string>;
}

export const EMPTY_DRAFT: Draft = { excluded: new Set(), expected: new Map() };

export const toggleCase = (draft: Draft, id: string): Draft => {
  const excluded = new Set(draft.excluded);
  if (!excluded.delete(id)) excluded.add(id);
  return { ...draft, excluded };
};

export const setExpected = (draft: Draft, id: string, text: string): Draft => ({
  ...draft,
  expected: new Map(draft.expected).set(id, text),
});

export const pickedCases = (built: readonly DatasetCase[], draft: Draft): DatasetCase[] =>
  built
    .filter((item) => !draft.excluded.has(item.id))
    .map((item) => ({ ...item, expected: draft.expected.get(item.id) ?? item.expected }));

/** A save replaces the whole list, so the dataset's current cases go first and the ticked new ones follow. */
export const revisionCases = (
  existing: readonly DatasetCase[],
  built: readonly DatasetCase[],
  draft: Draft,
): DatasetCase[] => [...existing, ...pickedCases(built, draft)];

export const casePrompt = (item: DatasetCase): string => {
  const users = item.messages.filter((message) => message.role === "user");
  return (users.at(-1) ?? item.messages.at(-1))?.content ?? "";
};

const callText = (call: DatasetToolCall): string => `${call.name}(${call.arguments})`;

export const caseReplySummary = (item: DatasetCase): string => {
  const calls = item.tool_calls.map(callText).join(", ");
  if (item.reply && calls) return `${item.reply} · ${calls}`;
  return item.reply || calls || "No reply recorded";
};
