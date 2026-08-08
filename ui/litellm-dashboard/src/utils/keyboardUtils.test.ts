import type { KeyboardEvent as ReactKeyboardEvent } from "react";
import { describe, expect, it } from "vitest";
import { isSubmitEnterKey } from "./keyboardUtils";

function makeEvent(overrides: {
  key?: string;
  shiftKey?: boolean;
  isComposing?: boolean;
  keyCode?: number;
}): ReactKeyboardEvent {
  return {
    key: overrides.key ?? "Enter",
    shiftKey: overrides.shiftKey ?? false,
    keyCode: overrides.keyCode ?? 13,
    nativeEvent: { isComposing: overrides.isComposing ?? false },
  } as ReactKeyboardEvent;
}

describe("isSubmitEnterKey", () => {
  it("returns true for plain Enter", () => {
    expect(isSubmitEnterKey(makeEvent({}))).toBe(true);
  });

  it("returns false for Shift+Enter", () => {
    expect(isSubmitEnterKey(makeEvent({ shiftKey: true }))).toBe(false);
  });

  it("returns false while IME is composing", () => {
    expect(isSubmitEnterKey(makeEvent({ isComposing: true }))).toBe(false);
  });

  it("returns false for legacy IME Process keyCode 229", () => {
    expect(isSubmitEnterKey(makeEvent({ keyCode: 229 }))).toBe(false);
  });

  it("returns false for non-Enter keys", () => {
    expect(isSubmitEnterKey(makeEvent({ key: "a" }))).toBe(false);
  });
});
