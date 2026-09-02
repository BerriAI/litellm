/**
 * State transitions for the classifier prompt editor, kept out of the component so they can be
 * asserted directly rather than through a render.
 */

export interface ClassifierPromptDraft {
  /** What the textarea shows. */
  text: string;
  /** The default rubric the proxy would send, used to decide whether the draft is a real override. */
  defaultPrompt: string;
}

/**
 * What to persist for a draft.
 *
 * A draft equal to the default is stored as undefined rather than as a copy of the rubric. Saving
 * the copy would silently pin the router to today's wording, so a later improvement to the built-in
 * rubric would reach every router except the ones whose operator opened the editor and changed
 * nothing. Whitespace-only is treated the same way, and matches the backend validator that rejects a
 * blank prompt instead of reading it as "use the default".
 */
export const resolveCustomPrompt = ({ text, defaultPrompt }: ClassifierPromptDraft): string | undefined => {
  const trimmed = text.trim();
  if (!trimmed) return undefined;
  if (trimmed === defaultPrompt.trim()) return undefined;
  return text;
};

/** Whether a saved config carries an operator-authored prompt rather than the built-in rubric. */
export const hasCustomPrompt = (systemPrompt: string | undefined): boolean => Boolean(systemPrompt?.trim());

/**
 * The text to open the editor with: the operator's prompt when they have one, otherwise the default
 * rubric so they edit the real thing rather than starting from an empty box.
 */
export const initialDraftText = (systemPrompt: string | undefined, defaultPrompt: string): string =>
  hasCustomPrompt(systemPrompt) ? (systemPrompt as string) : defaultPrompt;
