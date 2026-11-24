import { describe, expect, it } from "vitest";
import { reconciledTokenBreakdown, type TokenBreakdown } from "./tokenBreakdown";

const recorded: TokenBreakdown = {
  inputTokens: 80,
  outputTokens: 20,
  cacheReadTokens: 30,
  cacheWriteTokens: 10,
};
const unavailableCache = { ...recorded, cacheReadTokens: null, cacheWriteTokens: null };

describe("reconciledTokenBreakdown", () => {
  it("should preserve recorded counts when input, output and cache reconcile", () => {
    expect(reconciledTokenBreakdown(100, recorded)).toEqual(recorded);
  });

  it("should preserve known input and output but suppress cache when totals disagree", () => {
    expect(reconciledTokenBreakdown(150, recorded)).toEqual(unavailableCache);
  });

  it("should suppress overlapping cache counts that exceed input tokens", () => {
    const overlapping = { ...recorded, cacheReadTokens: 75 };
    expect(reconciledTokenBreakdown(100, overlapping)).toEqual(unavailableCache);
  });

  it("should keep recorded zero counts when the total is also zero", () => {
    const zero = { inputTokens: 0, outputTokens: 0, cacheReadTokens: 0, cacheWriteTokens: 0 };
    expect(reconciledTokenBreakdown(0, zero)).toEqual(zero);
  });

  it("should not fill missing cache counts with zero", () => {
    const missingCache = { ...recorded, cacheReadTokens: null };
    expect(reconciledTokenBreakdown(100, missingCache)).toEqual(unavailableCache);
  });

  it.each([null, Number.NaN, Number.POSITIVE_INFINITY, -1])(
    "should preserve known output but suppress cache when input is invalid (%s)",
    (inputTokens) => {
      const invalidInput = { ...recorded, inputTokens };
      const expected = { ...unavailableCache, inputTokens: null };
      expect(reconciledTokenBreakdown(100, invalidInput)).toEqual(expected);
    },
  );

  it.each([null, Number.NaN, Number.POSITIVE_INFINITY, -1])(
    "should preserve known input but suppress cache when output is invalid (%s)",
    (outputTokens) => {
      const invalidOutput = { ...recorded, outputTokens };
      const expected = { ...unavailableCache, outputTokens: null };
      expect(reconciledTokenBreakdown(100, invalidOutput)).toEqual(expected);
    },
  );

  it.each([Number.NaN, Number.POSITIVE_INFINITY, -1])(
    "should suppress invalid cache counts (%s)",
    (cacheWriteTokens) => {
      const invalidCache = { ...recorded, cacheWriteTokens };
      expect(reconciledTokenBreakdown(100, invalidCache)).toEqual(unavailableCache);
    },
  );

  it.each([Number.NaN, Number.POSITIVE_INFINITY, -1])(
    "should suppress cache when the total is invalid (%s)",
    (total) => {
      expect(reconciledTokenBreakdown(total, recorded)).toEqual(unavailableCache);
    },
  );
});
