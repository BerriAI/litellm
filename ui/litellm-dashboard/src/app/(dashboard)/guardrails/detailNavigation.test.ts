/* @vitest-environment jsdom */
import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { guardrailDetailHref, useGuardrailDetailRouting } from "./detailNavigation";

vi.mock("next/navigation", () => ({ useSearchParams: () => new URLSearchParams(window.location.search) }));
vi.mock("@/components/networking", () => ({ serverRootPath: "/" }));

describe("guardrailDetailHref", () => {
  it("links straight to the settings tab when asked for it", () => {
    expect(guardrailDetailHref("fee65a60", "settings")).toBe(
      "/ui/guardrails?guardrail=fee65a60&guardrail_tab=settings",
    );
  });

  it("omits the tab param for the default overview tab", () => {
    expect(guardrailDetailHref("fee65a60")).toBe("/ui/guardrails?guardrail=fee65a60");
  });
});

describe("useGuardrailDetailRouting", () => {
  beforeEach(() => {
    window.history.pushState(null, "", "/guardrails");
  });

  it("openGuardrail sets ?guardrail= via history.pushState (no full navigation)", () => {
    const spy = vi.spyOn(window.history, "pushState");
    const { result } = renderHook(() => useGuardrailDetailRouting());
    act(() => result.current.openGuardrail("fee65a60"));
    expect(spy).toHaveBeenCalledWith(null, "", expect.stringContaining("guardrail=fee65a60"));
    spy.mockRestore();
  });

  it("openGuardrail drops a stale tab param so the detail view opens on overview", () => {
    window.history.pushState(null, "", "/guardrails?guardrail=old&guardrail_tab=settings");
    const spy = vi.spyOn(window.history, "pushState");
    const { result } = renderHook(() => useGuardrailDetailRouting());
    act(() => result.current.openGuardrail("fee65a60"));
    const url = spy.mock.calls.at(-1)?.[2] as string;
    expect(url).toContain("guardrail=fee65a60");
    expect(url).not.toContain("guardrail_tab");
    spy.mockRestore();
  });

  it("close removes both guardrail params and keeps unrelated ones", () => {
    window.history.pushState(null, "", "/guardrails?tab=garden&guardrail=fee65a60&guardrail_tab=settings");
    const spy = vi.spyOn(window.history, "pushState");
    const { result } = renderHook(() => useGuardrailDetailRouting());
    act(() => result.current.close());
    const url = spy.mock.calls.at(-1)?.[2] as string;
    expect(url).toContain("tab=garden");
    expect(url).not.toContain("guardrail=");
    expect(url).not.toContain("guardrail_tab");
    spy.mockRestore();
  });

  it("reads the guardrail id and settings tab from the query string", () => {
    window.history.pushState(null, "", "/guardrails?guardrail=fee65a60&guardrail_tab=settings");
    const { result } = renderHook(() => useGuardrailDetailRouting());
    expect(result.current.guardrailId).toBe("fee65a60");
    expect(result.current.tab).toBe("settings");
  });

  it("falls back to the overview tab for an unknown tab value", () => {
    window.history.pushState(null, "", "/guardrails?guardrail=fee65a60&guardrail_tab=bogus");
    const { result } = renderHook(() => useGuardrailDetailRouting());
    expect(result.current.tab).toBe("overview");
  });

  it("guardrailId is null when no guardrail param is present", () => {
    const { result } = renderHook(() => useGuardrailDetailRouting());
    expect(result.current.guardrailId).toBeNull();
  });
});
