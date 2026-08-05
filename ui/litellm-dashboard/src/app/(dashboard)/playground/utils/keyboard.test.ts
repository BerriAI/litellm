import { describe, expect, it } from "vitest";
import type { KeyboardEvent } from "react";
import { isImeComposing } from "./keyboard";

const makeEvent = (overrides: { isComposing?: boolean; keyCode?: number }): KeyboardEvent<HTMLElement> =>
  ({
    keyCode: overrides.keyCode ?? 13,
    nativeEvent: { isComposing: overrides.isComposing ?? false } as globalThis.KeyboardEvent,
  }) as KeyboardEvent<HTMLElement>;

describe("isImeComposing", () => {
  it("returns true while the IME is composing", () => {
    expect(isImeComposing(makeEvent({ isComposing: true }))).toBe(true);
  });

  it("returns true for the legacy keyCode 229 composition signal", () => {
    expect(isImeComposing(makeEvent({ isComposing: false, keyCode: 229 }))).toBe(true);
  });

  it("returns false for a plain Enter press outside composition", () => {
    expect(isImeComposing(makeEvent({ isComposing: false, keyCode: 13 }))).toBe(false);
  });
});
