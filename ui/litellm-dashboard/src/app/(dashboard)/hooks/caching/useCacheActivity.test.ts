import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useCacheActivity, type CacheActivityParams } from "./useCacheActivity";

const useQueryMock = vi.fn();
vi.mock("@/lib/http/api", () => ({
  $api: { useQuery: (...args: unknown[]) => useQueryMock(...args) },
}));

const mockUseAuthorized = vi.fn();
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => mockUseAuthorized(),
}));

const params: CacheActivityParams = {
  startDate: "2026-07-20",
  endDate: "2026-07-27",
  keyAliases: ["my-key"],
  models: ["gpt-5.1"],
};

const lastCallOptions = (): { enabled: boolean } => {
  const calls = useQueryMock.mock.calls;
  return calls[calls.length - 1][3] as { enabled: boolean };
};

describe("useCacheActivity", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useQueryMock.mockReturnValue({ data: undefined });
    mockUseAuthorized.mockReturnValue({ accessToken: "test-access-token" });
  });

  it("queries GET /global/activity/cache_hits with dates and filters as query params", () => {
    renderHook(() => useCacheActivity(params));

    expect(useQueryMock).toHaveBeenCalledWith(
      "get",
      "/global/activity/cache_hits",
      {
        params: {
          query: {
            start_date: "2026-07-20",
            end_date: "2026-07-27",
            key_aliases: ["my-key"],
            models: ["gpt-5.1"],
          },
        },
      },
      expect.any(Object),
    );
  });

  it("enables the query when authorized and both dates are set", () => {
    renderHook(() => useCacheActivity(params));

    expect(lastCallOptions().enabled).toBe(true);
  });

  it("disables the query without an access token", () => {
    mockUseAuthorized.mockReturnValue({ accessToken: null });
    renderHook(() => useCacheActivity(params));

    expect(lastCallOptions().enabled).toBe(false);
  });

  it("disables the query while the date range is incomplete", () => {
    renderHook(() => useCacheActivity({ ...params, endDate: undefined }));

    expect(lastCallOptions().enabled).toBe(false);
  });
});
