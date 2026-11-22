import { describe, expect, it } from "vitest";
import { budgetUsageColor } from "./budgetUsageColor";

describe("budgetUsageColor", () => {
  it.each([
    [0, "#16A34A"],
    [59.99, "#16A34A"],
    [60, "#EAB308"],
    [79.99, "#EAB308"],
    [80, "#F97316"],
    [89.99, "#F97316"],
    [90, "#F05A2E"],
    [94.99, "#F05A2E"],
    [95, "#E5484D"],
    [100, "#E5484D"],
    [125, "#E5484D"],
  ] as const)("uses the expected budget band at %s%% usage", (percentage, color) => {
    expect(budgetUsageColor(percentage)).toBe(color);
  });
});
