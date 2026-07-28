/* @vitest-environment jsdom */
import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useKeyDetailRouting } from "./detailNavigation";

vi.mock("next/navigation", () => ({ useSearchParams: () => new URLSearchParams(window.location.search) }));

describe("useKeyDetailRouting", () => {
  beforeEach(() => {
    window.history.pushState(null, "", "/api-keys/");
  });

  it("openKey sets ?key= via history.pushState (no full navigation)", () => {
    const spy = vi.spyOn(window.history, "pushState");
    const { result } = renderHook(() => useKeyDetailRouting());
    act(() => result.current.openKey("88a145505dd6"));
    expect(spy).toHaveBeenCalledWith(null, "", expect.stringContaining("key=88a145505dd6"));
    spy.mockRestore();
  });

  it("openKey preserves unrelated query params like the legacy ?page=", () => {
    window.history.pushState(null, "", "/?page=api-keys");
    const spy = vi.spyOn(window.history, "pushState");
    const { result } = renderHook(() => useKeyDetailRouting());
    act(() => result.current.openKey("88a145505dd6"));
    const url = spy.mock.calls.at(-1)?.[2] as string;
    expect(url).toContain("page=api-keys");
    expect(url).toContain("key=88a145505dd6");
    spy.mockRestore();
  });

  it("close removes only the key param", () => {
    window.history.pushState(null, "", "/?page=api-keys&key=88a145505dd6");
    const spy = vi.spyOn(window.history, "pushState");
    const { result } = renderHook(() => useKeyDetailRouting());
    act(() => result.current.close());
    const url = spy.mock.calls.at(-1)?.[2] as string;
    expect(url).toContain("page=api-keys");
    expect(url).not.toContain("key=");
    spy.mockRestore();
  });

  it("exposes keyId from ?key=", () => {
    window.history.pushState(null, "", "/api-keys/?key=88a145505dd6");
    const { result } = renderHook(() => useKeyDetailRouting());
    expect(result.current.keyId).toBe("88a145505dd6");
  });

  it("keyId is null when no key param is present", () => {
    const { result } = renderHook(() => useKeyDetailRouting());
    expect(result.current.keyId).toBeNull();
  });
});
