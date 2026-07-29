/* @vitest-environment jsdom */
import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useDetailParam } from "./useDetailParam";

vi.mock("next/navigation", () => ({ useSearchParams: () => new URLSearchParams(window.location.search) }));

describe("useDetailParam", () => {
  beforeEach(() => {
    window.history.pushState(null, "", "/teams/");
  });

  it("open sets the param via history.pushState (no full navigation)", () => {
    const spy = vi.spyOn(window.history, "pushState");
    const { result } = renderHook(() => useDetailParam("team"));
    act(() => result.current.open("team-1"));
    expect(spy).toHaveBeenCalledWith(null, "", expect.stringContaining("team=team-1"));
    spy.mockRestore();
  });

  it("open preserves unrelated query params like the legacy ?page=", () => {
    window.history.pushState(null, "", "/?page=teams");
    const spy = vi.spyOn(window.history, "pushState");
    const { result } = renderHook(() => useDetailParam("team"));
    act(() => result.current.open("team-1"));
    const url = spy.mock.calls.at(-1)?.[2] as string;
    expect(url).toContain("page=teams");
    expect(url).toContain("team=team-1");
    spy.mockRestore();
  });

  it("close removes only its own param", () => {
    window.history.pushState(null, "", "/?page=teams&team=team-1");
    const spy = vi.spyOn(window.history, "pushState");
    const { result } = renderHook(() => useDetailParam("team"));
    act(() => result.current.close());
    const url = spy.mock.calls.at(-1)?.[2] as string;
    expect(url).toContain("page=teams");
    expect(url).not.toContain("team=");
    spy.mockRestore();
  });

  it("exposes the id from the param", () => {
    window.history.pushState(null, "", "/users/?user=user-7");
    const { result } = renderHook(() => useDetailParam("user"));
    expect(result.current.id).toBe("user-7");
  });

  it("id is null when the param is absent", () => {
    const { result } = renderHook(() => useDetailParam("org"));
    expect(result.current.id).toBeNull();
  });
});
