import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const useInfiniteQuery = vi.fn();
vi.mock("@/lib/http/api", () => ({ $api: { useInfiniteQuery: (...args: unknown[]) => useInfiniteQuery(...args) } }));

const mockUseAuthorized = vi.fn();
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => mockUseAuthorized(),
}));

import { useInfiniteSpendLogUsers } from "./useSpendLogUsers";

const WINDOW = { start_date: "2026-07-23 00:00:00", end_date: "2026-07-24 00:00:00" };

describe("useInfiniteSpendLogUsers", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseAuthorized.mockReturnValue({ accessToken: "test-token" });
  });

  it("calls the scoped spend-log user facet with the visible window", () => {
    renderHook(() => useInfiniteSpendLogUsers(WINDOW, 25, "alice"));

    const expectedQuery = {
      "filter[startTime][gte]": "2026-07-23 00:00:00",
      "filter[startTime][lte]": "2026-07-24 00:00:00",
      page_size: 25,
      q: "alice",
    };
    expect(useInfiniteQuery.mock.calls[0][1]).toBe("/management/v1/spend_logs/users");
    expect(useInfiniteQuery.mock.calls[0][2].params.query).toEqual(expectedQuery);
  });

  it("omits q when the search box is empty", () => {
    renderHook(() => useInfiniteSpendLogUsers(WINDOW, 50, ""));

    expect(useInfiniteQuery.mock.calls[0][2].params.query).not.toHaveProperty("q");
  });
});
