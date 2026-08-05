/* @vitest-environment jsdom */
import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useTeamDetailRouting } from "./detailNavigation";

vi.mock("next/navigation", () => ({ useSearchParams: () => new URLSearchParams(window.location.search) }));

describe("useTeamDetailRouting", () => {
  beforeEach(() => {
    window.history.pushState(null, "", "/teams/");
  });

  it("openTeam sets ?team= via history.pushState (no full navigation)", () => {
    const spy = vi.spyOn(window.history, "pushState");
    const { result } = renderHook(() => useTeamDetailRouting());
    act(() => result.current.openTeam("team-abc123"));
    expect(spy).toHaveBeenCalledWith(null, "", expect.stringContaining("team=team-abc123"));
    spy.mockRestore();
  });

  it("openTeam preserves unrelated query params", () => {
    window.history.pushState(null, "", "/teams/?foo=bar");
    const spy = vi.spyOn(window.history, "pushState");
    const { result } = renderHook(() => useTeamDetailRouting());
    act(() => result.current.openTeam("team-abc123"));
    const url = spy.mock.calls.at(-1)?.[2] as string;
    expect(url).toContain("foo=bar");
    expect(url).toContain("team=team-abc123");
    spy.mockRestore();
  });

  it("close removes only the team param", () => {
    window.history.pushState(null, "", "/teams/?foo=bar&team=team-abc123");
    const spy = vi.spyOn(window.history, "pushState");
    const { result } = renderHook(() => useTeamDetailRouting());
    act(() => result.current.close());
    const url = spy.mock.calls.at(-1)?.[2] as string;
    expect(url).toContain("foo=bar");
    expect(url).not.toContain("team=");
    spy.mockRestore();
  });

  it("exposes teamId from ?team=", () => {
    window.history.pushState(null, "", "/teams/?team=team-abc123");
    const { result } = renderHook(() => useTeamDetailRouting());
    expect(result.current.teamId).toBe("team-abc123");
  });

  it("teamId is null when no team param is present", () => {
    const { result } = renderHook(() => useTeamDetailRouting());
    expect(result.current.teamId).toBeNull();
  });
});
