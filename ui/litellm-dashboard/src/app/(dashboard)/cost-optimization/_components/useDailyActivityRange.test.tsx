import { renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

const mockUsePaginatedDailyActivity = vi.fn();

vi.mock("@/app/(dashboard)/usage/_components/hooks/usePaginatedDailyActivity", () => ({
  usePaginatedDailyActivity: (args: unknown) => {
    mockUsePaginatedDailyActivity(args);
    return { data: { results: [] }, loading: false, isFetchingMore: false };
  },
}));

vi.mock("@/components/networking", () => ({
  userDailyActivityCall: vi.fn(),
}));

import { useDailyActivityRange } from "./useDailyActivityRange";

const argsOfLastCall = () => mockUsePaginatedDailyActivity.mock.calls.at(-1)?.[0].args as unknown[];

describe("useDailyActivityRange", () => {
  it("queries every user's activity for an admin", () => {
    renderHook(() => useDailyActivityRange("test-token", "u1", "proxy_admin"));

    expect(argsOfLastCall()).toEqual(["test-token", expect.any(Date), expect.any(Date), null]);
  });

  it("scopes the query to the caller for a non-admin", () => {
    renderHook(() => useDailyActivityRange("test-token", "u1", "internal_user"));

    expect(argsOfLastCall()).toEqual(["test-token", expect.any(Date), expect.any(Date), "u1"]);
  });

  it("stays disabled until an access token is available", () => {
    renderHook(() => useDailyActivityRange(null, "u1", "proxy_admin"));

    expect(mockUsePaginatedDailyActivity).toHaveBeenLastCalledWith(expect.objectContaining({ enabled: false }));
  });
});
