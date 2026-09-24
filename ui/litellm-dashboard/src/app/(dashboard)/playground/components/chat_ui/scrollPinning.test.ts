import { describe, expect, it } from "vitest";
import { isPinnedToBottom, SCROLL_PIN_THRESHOLD_PX } from "./scrollPinning";

describe("isPinnedToBottom", () => {
  it("returns true when scrolled exactly to the bottom", () => {
    expect(isPinnedToBottom({ scrollTop: 900, scrollHeight: 1000, clientHeight: 100 })).toBe(true);
  });

  it("returns true when within the threshold of the bottom", () => {
    expect(isPinnedToBottom({ scrollTop: 880, scrollHeight: 1000, clientHeight: 100 })).toBe(true);
  });

  it("returns true when exactly at the threshold", () => {
    expect(
      isPinnedToBottom({
        scrollTop: 1000 - 100 - SCROLL_PIN_THRESHOLD_PX,
        scrollHeight: 1000,
        clientHeight: 100,
      }),
    ).toBe(true);
  });

  it("returns false one pixel past the threshold", () => {
    expect(
      isPinnedToBottom({
        scrollTop: 1000 - 100 - SCROLL_PIN_THRESHOLD_PX - 1,
        scrollHeight: 1000,
        clientHeight: 100,
      }),
    ).toBe(false);
  });

  it("returns false when scrolled far up", () => {
    expect(isPinnedToBottom({ scrollTop: 0, scrollHeight: 1000, clientHeight: 100 })).toBe(false);
  });
});
