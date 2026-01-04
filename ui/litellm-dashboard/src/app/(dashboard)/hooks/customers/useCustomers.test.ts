import { allEndUsersCall } from "@/components/networking";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import React, { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { Customer, CustomersResponse } from "./useCustomers";
import { useCustomers } from "./useCustomers";

// Mock the networking function
vi.mock("@/components/networking", () => ({
  allEndUsersCall: vi.fn(),
}));

// Mock useAuthorized hook - we can override this in individual tests
const mockUseAuthorized = vi.fn();
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => mockUseAuthorized(),
}));

// Import actual roles instead of mocking them

// Mock data
const mockCustomers: Customer[] = [
  {
    user_id: "customer-1",
    alias: "Test Customer 1",
    spend: 150.5,
    blocked: false,
    allowed_model_region: "us-east-1",
    default_model: "gpt-3.5-turbo",
    budget_id: "budget-1",
    litellm_budget_table: {
      budget_id: "budget-1",
      max_budget: 1000,
      soft_budget: 800,
      max_parallel_requests: 10,
      tpm_limit: 1000,
      rpm_limit: 100,
      model_max_budget: { "gpt-4": 500 },
      budget_duration: "monthly",
      budget_reset_at: "2024-02-01T00:00:00Z",
      created_at: "2024-01-01T00:00:00Z",
      created_by: "admin-1",
      updated_at: "2024-01-01T00:00:00Z",
      updated_by: "admin-1",
    },
  },
  {
    user_id: "customer-2",
    alias: null,
    spend: 0,
    blocked: true,
    allowed_model_region: null,
    default_model: null,
    budget_id: null,
    litellm_budget_table: null,
  },
];

const mockCustomersResponse: CustomersResponse = mockCustomers;

describe("useCustomers", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
        },
      },
    });

    // Reset all mocks
    vi.clearAllMocks();

    // Set default mock for useAuthorized (enabled state)
    mockUseAuthorized.mockReturnValue({
      accessToken: "test-access-token",
      userRole: "Admin",
      userId: "test-user-id",
      token: "test-token",
      userEmail: "test@example.com",
      premiumUser: false,
      disabledPersonalKeyCreation: null,
      showSSOBanner: false,
    });
  });

  const wrapper = ({ children }: { children: ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);

  it("should return customers data when query is successful", async () => {
    // Mock successful API call
    (allEndUsersCall as any).mockResolvedValue(mockCustomersResponse);

    const { result } = renderHook(() => useCustomers(), { wrapper });

    // Initially loading
    expect(result.current.isLoading).toBe(true);
    expect(result.current.data).toBeUndefined();

    // Wait for success
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual(mockCustomersResponse);
    expect(result.current.error).toBeNull();
    expect(allEndUsersCall).toHaveBeenCalledWith("test-access-token");
    expect(allEndUsersCall).toHaveBeenCalledTimes(1);
  });

  it("should handle error when allEndUsersCall fails", async () => {
    const errorMessage = "Failed to fetch customers";
    const testError = new Error(errorMessage);

    // Mock failed API call
    (allEndUsersCall as any).mockRejectedValue(testError);

    const { result } = renderHook(() => useCustomers(), { wrapper });

    // Initially loading
    expect(result.current.isLoading).toBe(true);

    // Wait for error
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(testError);
    expect(result.current.data).toBeUndefined();
    expect(allEndUsersCall).toHaveBeenCalledWith("test-access-token");
    expect(allEndUsersCall).toHaveBeenCalledTimes(1);
  });

  it("should not execute query when accessToken is missing", async () => {
    // Mock missing accessToken
    mockUseAuthorized.mockReturnValue({
      accessToken: null,
      userRole: "Admin",
      userId: "test-user-id",
      token: null,
      userEmail: "test@example.com",
      premiumUser: false,
      disabledPersonalKeyCreation: null,
      showSSOBanner: false,
    });

    const { result } = renderHook(() => useCustomers(), { wrapper });

    // Query should not execute
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);

    // API should not be called
    expect(allEndUsersCall).not.toHaveBeenCalled();
  });

  it("should not execute query when userRole is not an admin role", async () => {
    // Mock non-admin userRole
    mockUseAuthorized.mockReturnValue({
      accessToken: "test-access-token",
      userRole: "member", // Not in all_admin_roles
      userId: "test-user-id",
      token: "test-token",
      userEmail: "test@example.com",
      premiumUser: false,
      disabledPersonalKeyCreation: null,
      showSSOBanner: false,
    });

    const { result } = renderHook(() => useCustomers(), { wrapper });

    // Query should not execute
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);

    // API should not be called
    expect(allEndUsersCall).not.toHaveBeenCalled();
  });

  it("should not execute query when userRole is null", async () => {
    // Mock null userRole
    mockUseAuthorized.mockReturnValue({
      accessToken: "test-access-token",
      userRole: null,
      userId: "test-user-id",
      token: "test-token",
      userEmail: "test@example.com",
      premiumUser: false,
      disabledPersonalKeyCreation: null,
      showSSOBanner: false,
    });

    const { result } = renderHook(() => useCustomers(), { wrapper });

    // Query should not execute
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);

    // API should not be called
    expect(allEndUsersCall).not.toHaveBeenCalled();
  });

  it("should not execute query when userRole is empty string", async () => {
    // Mock empty string userRole
    mockUseAuthorized.mockReturnValue({
      accessToken: "test-access-token",
      userRole: "",
      userId: "test-user-id",
      token: "test-token",
      userEmail: "test@example.com",
      premiumUser: false,
      disabledPersonalKeyCreation: null,
      showSSOBanner: false,
    });

    const { result } = renderHook(() => useCustomers(), { wrapper });

    // Query should not execute
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);

    // API should not be called
    expect(allEndUsersCall).not.toHaveBeenCalled();
  });

  it("should not execute query when both accessToken and userRole are missing", async () => {
    // Mock both auth values missing
    mockUseAuthorized.mockReturnValue({
      accessToken: null,
      userRole: null,
      userId: "test-user-id",
      token: null,
      userEmail: "test@example.com",
      premiumUser: false,
      disabledPersonalKeyCreation: null,
      showSSOBanner: false,
    });

    const { result } = renderHook(() => useCustomers(), { wrapper });

    // Query should not execute
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);

    // API should not be called
    expect(allEndUsersCall).not.toHaveBeenCalled();
  });

  it("should execute query when accessToken is present and userRole is Admin", async () => {
    // Mock successful API call
    (allEndUsersCall as any).mockResolvedValue(mockCustomersResponse);

    // Ensure auth values are set (already done in beforeEach)
    const { result } = renderHook(() => useCustomers(), { wrapper });

    // Wait for query to execute
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(allEndUsersCall).toHaveBeenCalledWith("test-access-token");
    expect(allEndUsersCall).toHaveBeenCalledTimes(1);
  });

  it("should execute query when accessToken is present and userRole is proxy_admin", async () => {
    // Mock successful API call
    (allEndUsersCall as any).mockResolvedValue(mockCustomersResponse);

    // Mock proxy_admin role
    mockUseAuthorized.mockReturnValue({
      accessToken: "test-access-token",
      userRole: "proxy_admin",
      userId: "test-user-id",
      token: "test-token",
      userEmail: "test@example.com",
      premiumUser: false,
      disabledPersonalKeyCreation: null,
      showSSOBanner: false,
    });

    const { result } = renderHook(() => useCustomers(), { wrapper });

    // Wait for query to execute
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(allEndUsersCall).toHaveBeenCalledWith("test-access-token");
    expect(allEndUsersCall).toHaveBeenCalledTimes(1);
  });

  it("should return empty customers array when API returns empty data", async () => {
    // Mock API returning empty customers array
    (allEndUsersCall as any).mockResolvedValue([]);

    const { result } = renderHook(() => useCustomers(), { wrapper });

    // Wait for success
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual([]);
    expect(allEndUsersCall).toHaveBeenCalledWith("test-access-token");
  });

  it("should handle network timeout error", async () => {
    const timeoutError = new Error("Network timeout");

    // Mock network timeout
    (allEndUsersCall as any).mockRejectedValue(timeoutError);

    const { result } = renderHook(() => useCustomers(), { wrapper });

    // Wait for error
    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(timeoutError);
    expect(result.current.data).toBeUndefined();
  });
});
