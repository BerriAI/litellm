import type { KeyboardEvent as ReactKeyboardEvent } from "react";

/**
 * True when Enter should submit (not newline / not IME candidate confirm).
 * IME composition uses Enter to accept candidates; treating that as submit
 * breaks CJK input (Chinese / Japanese / Korean).
 */
export function isSubmitEnterKey(
  event: Pick<ReactKeyboardEvent, "key" | "shiftKey" | "nativeEvent" | "keyCode">,
): boolean {
  if (event.key !== "Enter" || event.shiftKey) {
    return false;
  }
  // During IME composition (or legacy keyCode 229 for Process key)
  if (event.nativeEvent.isComposing || event.keyCode === 229) {
    return false;
  }
  return true;
}
