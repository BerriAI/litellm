import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useFilterLogic } from "./filter_logic";
import { keyListCall } from "../networking";

vi.mock("../networking", () => ({
  keyListCall: vi.fn(),
}));

vi.mock("./filter_helpers", () => ({
  fetchAllTeams: vi.fn().mockResolvedValue([]),
  fetchAllOrganizations: vi.fn().mockResolvedValue([]),
}));

const mockKey = {
  token: "abc123",
  key_alias: "aaaaa",
  team_id: null,
  organization_id: null,
};

const defaultProps = {
  keys: [mockKey] as any[],
  teams: [],
  organizations: [],
};

const makeApiResponse = (overrides: { keys?: any[]; total_count?: number; total_pages?: number } = {}) => ({
  keys: overrides.keys ?? [mockKey],
  total_count: overrides.total_count ?? 1,
  current_page: 1,
  total_pages: overrides.total_pages ?? 1,
});

describe("useFilterLogic – stability", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(keyListCall).mockResolvedValue(makeApiResponse());
  });

  it("should not enter an infinite render loop when keys prop is re-rendered with a new empty-array reference", async () => {
    // Regression: callers that write `keys?.keys || []` produce a fresh `[]`
    // on every render (when keys is undefined/null).  The useEffect([keys, filters])
    // must not treat every new-reference empty array as a change that requires
    // another setFilteredKeys call, which would re-render the consumer, which
    // would produce yet another new `[]`, ad infinitum.
    const { result, rerender } = renderHook(
      ({ keys }) => useFilterLogic({ keys, teams: [], organizations: [] }),
      { initialProps: { keys: [] as any[] } },
    );

    // Simulate the || [] pattern: each rerender gets a brand-new [] literal
    act(() => {
      rerender({ keys: [] });
      rerender({ keys: [] });
      rerender({ keys: [] });
    });

    // If we reach here the hook did not loop.
    // filteredKeys should reflect the empty input.
    expect(result.current.filteredKeys).toEqual([]);
  });

  it("should update filteredKeys when keys prop changes from empty to populated", async () => {
    const { result, rerender } = renderHook(
      ({ keys }) => useFilterLogic({ keys, teams: [], organizations: [] }),
      { initialProps: { keys: [] as any[] } },
    );

    expect(result.current.filteredKeys).toEqual([]);

    act(() => {
      rerender({ keys: [mockKey as any] });
    });

    expect(result.current.filteredKeys).toHaveLength(1);
    expect(result.current.filteredKeys[0]).toBe(mockKey);
  });
});

describe("useFilterLogic – filteredTotalCount", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(keyListCall).mockResolvedValue(makeApiResponse({ total_count: 509, total_pages: 11 }));
  });

  it("should expose filteredTotalCount as null before any filter search runs", () => {
    const { result } = renderHook(() => useFilterLogic(defaultProps));

    expect(result.current.filteredTotalCount).toBeNull();
  });

  it("should set filteredTotalCount to the API total_count after a Key Alias filter is applied", async () => {
    vi.mocked(keyListCall).mockResolvedValue(makeApiResponse({ keys: [mockKey], total_count: 1, total_pages: 1 }));

    const { result } = renderHook(() => useFilterLogic(defaultProps));

    act(() => {
      result.current.handleFilterChange({ "Key Alias": "aaaaa" });
    });

    await waitFor(() => {
      expect(result.current.filteredTotalCount).toBe(1);
    }, { timeout: 500 });
  });

  it("should reflect the filtered total_count even when it differs from the full key count", async () => {
    vi.mocked(keyListCall).mockResolvedValue(makeApiResponse({ total_count: 7, total_pages: 1 }));

    const { result } = renderHook(() => useFilterLogic(defaultProps));

    act(() => {
      result.current.handleFilterChange({ "Team ID": "team-x" });
    });

    await waitFor(() => {
      expect(result.current.filteredTotalCount).toBe(7);
    }, { timeout: 500 });
  });

  it("should reset filteredTotalCount to null when handleFilterReset is called", async () => {
    vi.mocked(keyListCall).mockResolvedValue(makeApiResponse({ total_count: 1 }));

    const { result } = renderHook(() => useFilterLogic(defaultProps));

    act(() => {
      result.current.handleFilterChange({ "Key Alias": "aaaaa" });
    });

    await waitFor(() => {
      expect(result.current.filteredTotalCount).toBe(1);
    }, { timeout: 500 });

    act(() => {
      result.current.handleFilterReset();
    });

    // filteredTotalCount resets synchronously before the debounced reset search completes
    expect(result.current.filteredTotalCount).toBeNull();
  });

  it("should pass the Key Alias value to keyListCall", async () => {
    vi.mocked(keyListCall).mockResolvedValue(makeApiResponse({ total_count: 2 }));

    const { result } = renderHook(() => useFilterLogic(defaultProps));

    act(() => {
      result.current.handleFilterChange({ "Key Alias": "my-alias" });
    });

    await waitFor(() => {
      expect(keyListCall).toHaveBeenCalledWith(
        expect.any(String), // accessToken
        null,              // organizationID (empty → null)
        null,              // teamID (empty → null)
        "my-alias",        // selectedKeyAlias ← the filter value
        null,              // userID
        null,              // keyHash
        1,                 // page (resets to 1 on filter change)
        expect.any(Number),// pageSize (defaultPageSize)
        expect.anything(), // sortBy
        expect.anything(), // sortOrder
      );
    }, { timeout: 500 });
  });

  it("should not update filteredTotalCount when keyListCall throws", async () => {
    vi.mocked(keyListCall).mockRejectedValue(new Error("Network error"));

    const { result } = renderHook(() => useFilterLogic(defaultProps));

    act(() => {
      result.current.handleFilterChange({ "Key Alias": "bad-alias" });
    });

    await waitFor(() => {
      expect(keyListCall).toHaveBeenCalled();
    }, { timeout: 500 });

    expect(result.current.filteredTotalCount).toBeNull();
  });

  it("should not trigger a debounced search when skipDebounce is true", async () => {
    const { result } = renderHook(() => useFilterLogic(defaultProps));

    act(() => {
      result.current.handleFilterChange({ "Sort By": "spend", "Sort Order": "asc" }, true);
    });

    await new Promise((resolve) => setTimeout(resolve, 350));

    expect(keyListCall).not.toHaveBeenCalled();
    expect(result.current.filteredTotalCount).toBeNull();
  });
});
