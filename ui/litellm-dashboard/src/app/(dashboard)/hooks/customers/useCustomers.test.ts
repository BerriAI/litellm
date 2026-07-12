import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import React, { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useCustomers, type EndUser } from "./useCustomers";

const mockGet = vi.fn();
vi.mock("@/lib/http/api", () => ({
  fetchClient: { GET: (...args: unknown[]) => mockGet(...args) },
}));

const mockUseAuthorized = vi.fn();
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => mockUseAuthorized(),
}));

const mockCustomers: EndUser[] = [
  { user_id: "customer-1", alias: "Test Customer 1", spend: 150.5, blocked: false },
  { user_id: "customer-2", alias: null, spend: 0, blocked: true },
];

const authorized = {
  accessToken: "test-access-token",
  userRole: "Admin",
  userId: "test-user-id",
  token: "test-token",
  userEmail: "test@example.com",
  premiumUser: false,
  disabledPersonalKeyCreation: null,
  showSSOBanner: false,
};

describe("useCustomers", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    vi.clearAllMocks();
    mockUseAuthorized.mockReturnValue(authorized);
  });

  const wrapper = ({ children }: { children: ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);

  it("fetches /customer/list and returns the typed list on success", async () => {
    mockGet.mockResolvedValue({ data: mockCustomers });

    const { result } = renderHook(() => useCustomers(), { wrapper });

    expect(result.current.isLoading).toBe(true);

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual(mockCustomers);
    expect(mockGet).toHaveBeenCalledWith("/customer/list");
    expect(mockGet).toHaveBeenCalledTimes(1);
  });

  it("surfaces an error when the request rejects", async () => {
    const testError = new Error("Failed to fetch customers");
    mockGet.mockRejectedValue(testError);

    const { result } = renderHook(() => useCustomers(), { wrapper });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(testError);
    expect(result.current.data).toBeUndefined();
  });

  it("falls back to an empty list when the response has no body", async () => {
    mockGet.mockResolvedValue({ data: undefined });

    const { result } = renderHook(() => useCustomers(), { wrapper });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual([]);
  });

  it("does not fetch when the access token is missing", () => {
    mockUseAuthorized.mockReturnValue({ ...authorized, accessToken: null, token: null });

    const { result } = renderHook(() => useCustomers(), { wrapper });

    expect(result.current.isFetched).toBe(false);
    expect(mockGet).not.toHaveBeenCalled();
  });

  it("does not fetch when the user is not an admin", () => {
    mockUseAuthorized.mockReturnValue({ ...authorized, userRole: "member" });

    const { result } = renderHook(() => useCustomers(), { wrapper });

    expect(result.current.isFetched).toBe(false);
    expect(mockGet).not.toHaveBeenCalled();
  });
});
