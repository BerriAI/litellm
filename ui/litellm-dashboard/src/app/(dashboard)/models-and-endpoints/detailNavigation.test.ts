/* @vitest-environment jsdom */
import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useModelDetailRouting } from "./detailNavigation";

// The detail overlay is driven by ?model=/?team= on the current path. Under the
// /ui static mount a router.push to the same path (query-only change) is a no-op,
// so navigation goes through history.pushState (client-side, no full reload).
vi.mock("next/navigation", () => ({ useSearchParams: () => new URLSearchParams(window.location.search) }));

describe("useModelDetailRouting", () => {
  beforeEach(() => {
    window.history.pushState(null, "", "/models-and-endpoints/");
  });

  it("openModel sets ?model= via history.pushState (no full navigation)", () => {
    const spy = vi.spyOn(window.history, "pushState");
    const { result } = renderHook(() => useModelDetailRouting());
    act(() => result.current.openModel("abc-1"));
    expect(spy).toHaveBeenCalledWith(null, "", expect.stringContaining("model=abc-1"));
    spy.mockRestore();
  });

  it("openTeam sets ?team= and drops any model param", () => {
    window.history.pushState(null, "", "/models-and-endpoints/?model=abc-1");
    const spy = vi.spyOn(window.history, "pushState");
    const { result } = renderHook(() => useModelDetailRouting());
    act(() => result.current.openTeam("team-9"));
    const url = spy.mock.calls.at(-1)?.[2] as string;
    expect(url).toContain("team=team-9");
    expect(url).not.toContain("model=");
    spy.mockRestore();
  });

  it("close removes both model and team params", () => {
    window.history.pushState(null, "", "/models-and-endpoints/?model=abc-1");
    const spy = vi.spyOn(window.history, "pushState");
    const { result } = renderHook(() => useModelDetailRouting());
    act(() => result.current.close());
    const url = spy.mock.calls.at(-1)?.[2] as string;
    expect(url).not.toContain("model=");
    expect(url).not.toContain("team=");
    spy.mockRestore();
  });

  it("reads modelId and teamId from the query string", () => {
    window.history.pushState(null, "", "/models-and-endpoints/?model=xyz");
    const { result } = renderHook(() => useModelDetailRouting());
    expect(result.current.modelId).toBe("xyz");
    expect(result.current.teamId).toBeNull();
  });
});
