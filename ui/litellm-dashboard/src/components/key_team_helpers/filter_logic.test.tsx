import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useFilterLogic } from "./filter_logic";

vi.mock("./filter_helpers", () => ({
  fetchAllTeams: vi.fn().mockResolvedValue([]),
  fetchAllOrganizations: vi.fn().mockResolvedValue([]),
}));

const defaultProps = {
  teams: [],
  organizations: [],
};

const DEFAULT_FILTERS = {
  "Team ID": "",
  "Organization ID": "",
  "Key Alias": "",
  "User ID": "",
  "Sort By": "created_at",
  "Sort Order": "desc",
};

describe("useFilterLogic – filter state management", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("starts with default (empty) filter values", () => {
    const { result } = renderHook(() => useFilterLogic(defaultProps));

    expect(result.current.filters).toEqual(DEFAULT_FILTERS);
  });

  it("updates the Key Alias filter when handleFilterChange is called", () => {
    const { result } = renderHook(() => useFilterLogic(defaultProps));

    act(() => {
      result.current.handleFilterChange({ "Key Alias": "my-alias" });
    });

    expect(result.current.filters["Key Alias"]).toBe("my-alias");
  });

  it("preserves filter state across re-renders (regression: bug where results reset on remount)", () => {
    const { result, rerender } = renderHook(() => useFilterLogic(defaultProps));

    act(() => {
      result.current.handleFilterChange({ "Key Alias": "keep-me" });
    });
    expect(result.current.filters["Key Alias"]).toBe("keep-me");

    // Re-render with the same props; filter must remain applied so downstream
    // useKeys is still invoked with the alias instead of falling back to "all".
    rerender();
    expect(result.current.filters["Key Alias"]).toBe("keep-me");
  });

  it("resets filters to defaults when handleFilterReset is called", () => {
    const { result } = renderHook(() => useFilterLogic(defaultProps));

    act(() => {
      result.current.handleFilterChange({ "Team ID": "team-x", "Key Alias": "abc" });
    });
    expect(result.current.filters["Team ID"]).toBe("team-x");
    expect(result.current.filters["Key Alias"]).toBe("abc");

    act(() => {
      result.current.handleFilterReset();
    });
    expect(result.current.filters).toEqual(DEFAULT_FILTERS);
  });

  it("handleFilterChange replaces the entire filter set (missing keys become empty)", () => {
    const { result } = renderHook(() => useFilterLogic(defaultProps));

    act(() => {
      result.current.handleFilterChange({ "Team ID": "team-1", "Key Alias": "alpha" });
    });

    act(() => {
      result.current.handleFilterChange({ "User ID": "user-1" });
    });

    expect(result.current.filters).toEqual({
      ...DEFAULT_FILTERS,
      "User ID": "user-1",
    });
  });
});
