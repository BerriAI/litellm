/* @vitest-environment jsdom */
import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useOrgDetailRouting } from "./detailNavigation";

vi.mock("next/navigation", () => ({ useSearchParams: () => new URLSearchParams(window.location.search) }));

describe("useOrgDetailRouting", () => {
  beforeEach(() => {
    window.history.pushState(null, "", "/organizations/");
  });

  it("openOrg sets ?org= via history.pushState (no full navigation)", () => {
    const spy = vi.spyOn(window.history, "pushState");
    const { result } = renderHook(() => useOrgDetailRouting());
    act(() => result.current.openOrg("org-abc123"));
    expect(spy).toHaveBeenCalledWith(null, "", expect.stringContaining("org=org-abc123"));
    spy.mockRestore();
  });

  it("openOrg preserves unrelated query params", () => {
    window.history.pushState(null, "", "/organizations/?foo=bar");
    const spy = vi.spyOn(window.history, "pushState");
    const { result } = renderHook(() => useOrgDetailRouting());
    act(() => result.current.openOrg("org-abc123"));
    const url = spy.mock.calls.at(-1)?.[2] as string;
    expect(url).toContain("foo=bar");
    expect(url).toContain("org=org-abc123");
    spy.mockRestore();
  });

  it("close removes only the org param", () => {
    window.history.pushState(null, "", "/organizations/?foo=bar&org=org-abc123");
    const spy = vi.spyOn(window.history, "pushState");
    const { result } = renderHook(() => useOrgDetailRouting());
    act(() => result.current.close());
    const url = spy.mock.calls.at(-1)?.[2] as string;
    expect(url).toContain("foo=bar");
    expect(url).not.toContain("org=");
    spy.mockRestore();
  });

  it("exposes orgId from ?org=", () => {
    window.history.pushState(null, "", "/organizations/?org=org-abc123");
    const { result } = renderHook(() => useOrgDetailRouting());
    expect(result.current.orgId).toBe("org-abc123");
  });

  it("orgId is null when no org param is present", () => {
    const { result } = renderHook(() => useOrgDetailRouting());
    expect(result.current.orgId).toBeNull();
  });
});
