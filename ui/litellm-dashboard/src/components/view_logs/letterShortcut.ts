const isTypingTarget = (target: EventTarget | null): boolean =>
  target instanceof HTMLElement &&
  target.matches("input, textarea, select, [contenteditable='true'], [role='combobox']");

const isModified = (event: KeyboardEvent): boolean => event.metaKey || event.ctrlKey || event.altKey;

/** Single-key drawer shortcuts (J, K, Esc) yield to typing and to modified presses such as Cmd+J. */
export const ignoresLetterShortcut = (event: KeyboardEvent): boolean =>
  isModified(event) || isTypingTarget(event.target);
