import { describe, expect, it } from "vitest";
import { getCapabilityBadgeClassName, getCapabilityBadgeColor } from "./capabilityColors";

describe("getCapabilityBadgeColor", () => {
  it("maps the same capability to the same color regardless of order", () => {
    const a = getCapabilityBadgeColor("supports_function_calling");
    const b = getCapabilityBadgeColor("supports_function_calling");
    expect(a).toBe(b);
    expect(a).toBe("blue");
  });

  it("uses different stable colors for different known capabilities", () => {
    expect(getCapabilityBadgeColor("supports_function_calling")).not.toBe(getCapabilityBadgeColor("supports_vision"));
  });

  it("is stable for unknown capability keys (hash-based)", () => {
    const key = "supports_custom_feature_xyz";
    expect(getCapabilityBadgeColor(key)).toBe(getCapabilityBadgeColor(key));
  });
});

describe("getCapabilityBadgeClassName", () => {
  it("returns matching tailwind classes for a capability", () => {
    const className = getCapabilityBadgeClassName("supports_vision");
    expect(className).toContain("purple");
  });
});
