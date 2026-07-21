import { renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { useIsPtuCostAttributionEnabled } from "./useIsPtuCostAttributionEnabled";

const mockUseUISettings = vi.hoisted(() => vi.fn());
vi.mock("@/app/(dashboard)/hooks/uiSettings/useUISettings", () => ({
  useUISettings: mockUseUISettings,
}));

describe("useIsPtuCostAttributionEnabled", () => {
  it("reads enable_ptu_cost_attribution from /get/ui_settings values (regression: v1 read the wrong endpoint)", () => {
    mockUseUISettings.mockReturnValue({
      data: { values: { enable_ptu_cost_attribution: true } },
      isLoading: false,
    });
    const { result } = renderHook(() => useIsPtuCostAttributionEnabled());
    expect(result.current.enabled).toBe(true);
    expect(result.current.isLoading).toBe(false);
  });

  it("returns enabled=false when the flag is not set in ui_settings values", () => {
    mockUseUISettings.mockReturnValue({ data: { values: {} }, isLoading: false });
    const { result } = renderHook(() => useIsPtuCostAttributionEnabled());
    expect(result.current.enabled).toBe(false);
  });

  it("returns enabled=false when ui_settings data is undefined (still loading or missing)", () => {
    mockUseUISettings.mockReturnValue({ data: undefined, isLoading: true });
    const { result } = renderHook(() => useIsPtuCostAttributionEnabled());
    expect(result.current.enabled).toBe(false);
    expect(result.current.isLoading).toBe(true);
  });

  it("returns enabled=false when the flag value is explicitly false", () => {
    mockUseUISettings.mockReturnValue({
      data: { values: { enable_ptu_cost_attribution: false } },
      isLoading: false,
    });
    const { result } = renderHook(() => useIsPtuCostAttributionEnabled());
    expect(result.current.enabled).toBe(false);
  });
});
