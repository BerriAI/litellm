import { renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { useCurrency } from "./useCurrency";
import { useUISettings } from "./useUISettings";

vi.mock("./useUISettings", () => ({ useUISettings: vi.fn() }));

describe("useCurrency", () => {
  it("defaults to USD while loading or with invalid settings", () => {
    vi.mocked(useUISettings).mockReturnValue({ data: undefined } as ReturnType<typeof useUISettings>);
    const { result, rerender } = renderHook(() => useCurrency());
    expect(result.current).toEqual({ code: "USD", usd_rate: 1 });
    vi.mocked(useUISettings).mockReturnValue({
      data: { currency: { code: "EUR", usd_rate: 0 } },
    } as unknown as ReturnType<typeof useUISettings>);
    rerender();
    expect(result.current).toEqual({ code: "USD", usd_rate: 1 });
  });

  it("reads configured currency", () => {
    vi.mocked(useUISettings).mockReturnValue({
      data: { currency: { code: "EUR", usd_rate: 0.92 } },
    } as unknown as ReturnType<typeof useUISettings>);
    expect(renderHook(() => useCurrency()).result.current).toEqual({ code: "EUR", usd_rate: 0.92 });
  });
});
