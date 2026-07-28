import { describe, expect, it } from "vitest";
import { CAPABILITY_COLOR_PALETTE, getCapabilityBadgeClassName, getCapabilityColor } from "./capabilityColors";

const ANTD_TAG_PRESET_COLORS = new Set([
  "blue",
  "purple",
  "cyan",
  "green",
  "magenta",
  "pink",
  "red",
  "orange",
  "yellow",
  "volcano",
  "geekblue",
  "lime",
  "gold",
]);

describe("getCapabilityColor", () => {
  it("returns the same color for the same feature across different rows / order", () => {
    expect(getCapabilityColor("supports_function_calling")).toBe(getCapabilityColor("Function Calling"));
    expect(getCapabilityColor("supports_vision")).toBe(getCapabilityColor("Vision"));
    expect(getCapabilityColor("supports_reasoning")).toBe(getCapabilityColor("Reasoning"));
  });

  it("keeps Function Calling the same color whether it is first or second in a list", () => {
    const featuresA = ["supports_function_calling", "supports_reasoning"];
    const featuresB = ["supports_vision", "supports_function_calling"];
    expect(getCapabilityColor(featuresA[0])).toBe(getCapabilityColor(featuresB[1]));
    expect(getCapabilityColor(featuresA[0])).not.toBe(getCapabilityColor(featuresB[0]));
  });

  it("uses stable semantic colors for common capabilities", () => {
    expect(getCapabilityColor("supports_function_calling")).toBe("blue");
    expect(getCapabilityColor("supports_vision")).toBe("purple");
    expect(getCapabilityColor("supports_reasoning")).toBe("orange");
  });

  it("falls back to a deterministic palette color for unknown capabilities", () => {
    expect(getCapabilityColor("supports_custom_widget")).toBe(getCapabilityColor("Custom Widget"));
  });

  it("only returns colors that Ant Design Tag treats as presets", () => {
    for (const color of CAPABILITY_COLOR_PALETTE) {
      expect(ANTD_TAG_PRESET_COLORS.has(color)).toBe(true);
    }
    expect(ANTD_TAG_PRESET_COLORS.has(getCapabilityColor("supports_response_schema"))).toBe(true);
    expect(ANTD_TAG_PRESET_COLORS.has(getCapabilityColor("supports_system_messages"))).toBe(true);
    expect(ANTD_TAG_PRESET_COLORS.has(getCapabilityColor("supports_custom_widget"))).toBe(true);
  });
});

describe("getCapabilityBadgeClassName", () => {
  it("returns a non-empty class for known capabilities", () => {
    expect(getCapabilityBadgeClassName("supports_vision")).toContain("bg-purple-100");
  });
});
