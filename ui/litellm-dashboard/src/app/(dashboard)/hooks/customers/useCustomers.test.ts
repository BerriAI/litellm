import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useCustomers, type EndUser } from "./useCustomers";

const useQueryMock = vi.fn();
vi.mock("@/lib/http/api", () => ({
  $api: { useQuery: (...args: unknown[]) => useQueryMock(...args) },
}));

const mockUseAuthorized = vi.fn();
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => mockUseAuthorized(),
}));

const authorized = { accessToken: "test-access-token", userRole: "Admin" };

type QueryOptions = { enabled: boolean; select: (data: EndUser[] | undefined) => EndUser[] };

const lastCallOptions = (): QueryOptions => {
  const calls = useQueryMock.mock.calls;
  return calls[calls.length - 1][3] as QueryOptions;
};

describe("useCustomers", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useQueryMock.mockReturnValue({ data: [] });
    mockUseAuthorized.mockReturnValue(authorized);
  });

  it("queries GET /customer/list with a derived key (no hand-written queryKey)", () => {
    renderHook(() => useCustomers());
    expect(useQueryMock).toHaveBeenCalledWith("get", "/customer/list", {}, expect.any(Object));
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

  it("selects an empty list when the response body is missing", () => {
    renderHook(() => useCustomers());
    expect(lastCallOptions().select(undefined)).toEqual([]);
  });

  it("selects the customer list through unchanged", () => {
    const customers: EndUser[] = [
      { user_id: "customer-1", alias: "Test Customer 1", spend: 150.5, blocked: false },
      { user_id: "customer-2", alias: null, spend: 0, blocked: true },
    ];
    renderHook(() => useCustomers());
    expect(lastCallOptions().select(customers)).toEqual(customers);
  });
});
