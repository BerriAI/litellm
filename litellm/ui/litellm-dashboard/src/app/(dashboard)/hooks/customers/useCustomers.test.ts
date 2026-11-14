import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useCustomers, type EndUser } from "./useCustomers";

const useInfiniteQueryMock = vi.fn();
vi.mock("@/lib/http/api", () => ({
  $api: { useInfiniteQuery: (...args: unknown[]) => useInfiniteQueryMock(...args) },
}));

const mockUseAuthorized = vi.fn();
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => mockUseAuthorized(),
}));

const authorized = { accessToken: "test-access-token", userRole: "Admin" };

type QueryOptions = {
  enabled: boolean;
  getNextPageParam: (lastPage: EndUser[], pages: EndUser[][], lastPageParam: number) => number | undefined;
};

const lastCallOptions = (): QueryOptions => {
  const calls = useInfiniteQueryMock.mock.calls;
  return calls[calls.length - 1][3] as QueryOptions;
};

describe("useCustomers", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useInfiniteQueryMock.mockReturnValue({ data: [] });
    mockUseAuthorized.mockReturnValue(authorized);
  });

  it("includes page size and search in the request identity", () => {
    renderHook(() => useCustomers(20, "customer-50"));
    expect(useInfiniteQueryMock).toHaveBeenCalledWith(
      "get",
      "/customer/list",
      { params: { query: { page_size: 20, search: "customer-50" } } },
      expect.any(Object),
    );
  });

  it("enables the query only for an admin holding an access token", () => {
    renderHook(() => useCustomers());
    expect(lastCallOptions().enabled).toBe(true);
  });

  it("disables the query when the access token is missing", () => {
    mockUseAuthorized.mockReturnValue({ ...authorized, accessToken: null });
    renderHook(() => useCustomers());
    expect(lastCallOptions().enabled).toBe(false);
  });

  it("disables the query for a non-admin role", () => {
    mockUseAuthorized.mockReturnValue({ ...authorized, userRole: "member" });
    renderHook(() => useCustomers());
    expect(lastCallOptions().enabled).toBe(false);
  });

  it("offers the next page only when the current page is full", () => {
    const customers: EndUser[] = [
      { user_id: "customer-1", alias: "Test Customer 1", spend: 150.5, blocked: false },
      { user_id: "customer-2", alias: null, spend: 0, blocked: true },
    ];
    renderHook(() => useCustomers(2));
    expect(lastCallOptions().getNextPageParam(customers, [customers], 1)).toBe(2);
    expect(lastCallOptions().getNextPageParam(customers.slice(0, 1), [customers], 2)).toBeUndefined();
    expect(lastCallOptions().getNextPageParam([], [customers], 3)).toBeUndefined();
  });
});
