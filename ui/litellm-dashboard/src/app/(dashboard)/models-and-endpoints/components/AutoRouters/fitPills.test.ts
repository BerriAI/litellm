import { describe, expect, it } from "vitest";

import { fitPills, pillWidth } from "./fitPills";

const TARGETS = ["anthropic-sonnet-4-6", "gpt-4o-mini", "anthropic-opus-4-6", "voyage-4-large"];

describe("fitPills", () => {
  it("keeps everything on one row when it all fits", () => {
    const wide = TARGETS.reduce((total, label) => total + pillWidth(label) + 4, 0) + 40;
    expect(fitPills(TARGETS, wide)).toEqual({ visible: TARGETS, overflow: 0 });
  });

  it("shows more pills as the column gets wider", () => {
    const narrow = fitPills(TARGETS, 200);
    const wider = fitPills(TARGETS, 420);

    expect(narrow.visible.length).toBeLessThan(wider.visible.length);
    expect(narrow.visible.length + narrow.overflow).toBe(TARGETS.length);
    expect(wider.visible.length + wider.overflow).toBe(TARGETS.length);
  });

  it("reserves room for the +N counter so the row never overflows", () => {
    const { visible } = fitPills(TARGETS, 220);
    const used = visible.reduce((total, label, index) => total + pillWidth(label) + (index === 0 ? 0 : 4), 0);
    // 28px counter + its 4px gap must still fit alongside the visible pills.
    expect(used + 32).toBeLessThanOrEqual(220);
  });

  it("always shows at least one pill, even when a single name is wider than the column", () => {
    expect(fitPills(["an-extremely-long-deployment-name-that-never-fits"], 40)).toEqual({
      visible: ["an-extremely-long-deployment-name-that-never-fits"],
      overflow: 0,
    });
  });

  it("shows one pill before the first measurement rather than flashing every pill", () => {
    expect(fitPills(TARGETS, 0)).toEqual({ visible: [TARGETS[0]], overflow: 3 });
  });

  it("handles an empty target list", () => {
    expect(fitPills([], 300)).toEqual({ visible: [], overflow: 0 });
  });
});
