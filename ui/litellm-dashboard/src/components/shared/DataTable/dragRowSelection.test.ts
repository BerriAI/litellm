import { describe, expect, it } from "vitest";

import { autoScrollStep } from "./dragRowSelection";

const TOP = 100;
const BOTTOM = 600;

describe("autoScrollStep", () => {
  it("does not scroll while the pointer is away from both edges", () => {
    expect(autoScrollStep(350, TOP, BOTTOM)).toBe(0);
    expect(autoScrollStep(TOP + 40, TOP, BOTTOM)).toBe(0);
    expect(autoScrollStep(BOTTOM - 40, TOP, BOTTOM)).toBe(0);
  });

  it("scrolls up near the top edge and down near the bottom edge", () => {
    expect(autoScrollStep(TOP + 39, TOP, BOTTOM)).toBeLessThan(0);
    expect(autoScrollStep(BOTTOM - 39, TOP, BOTTOM)).toBeGreaterThan(0);
  });

  it("scrolls faster the closer the pointer is to the edge", () => {
    expect(autoScrollStep(TOP + 5, TOP, BOTTOM)).toBeLessThan(autoScrollStep(TOP + 30, TOP, BOTTOM));
    expect(autoScrollStep(BOTTOM - 5, TOP, BOTTOM)).toBeGreaterThan(autoScrollStep(BOTTOM - 30, TOP, BOTTOM));
  });

  it("caps the speed once the pointer reaches or passes the edge", () => {
    expect(autoScrollStep(TOP - 300, TOP, BOTTOM)).toBe(autoScrollStep(TOP, TOP, BOTTOM));
    expect(autoScrollStep(BOTTOM + 300, TOP, BOTTOM)).toBe(autoScrollStep(BOTTOM, TOP, BOTTOM));
    expect(autoScrollStep(BOTTOM, TOP, BOTTOM)).toBe(-autoScrollStep(TOP, TOP, BOTTOM));
  });
});
