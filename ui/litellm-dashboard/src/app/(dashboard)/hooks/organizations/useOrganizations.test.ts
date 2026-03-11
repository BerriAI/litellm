import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React, { ReactNode } from "react";
import { useOrganizations } from "./useOrganizations";
import { organizationListCall } from "@/components/networking";
import type { Organization } from "@/components/networking";

// Mock the networking function
vi.mock("@/components/networking", () => ({
  organizationListCall: vi.fn(),
}));

// Mock useAuthorized hook - we can override this in individual tests
const mockUseAuthorized = vi.fn();
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => mockUseAuthorized(),
}));

// Mock data
const mockOrganizations: Organization[] = [
  {
    organization_id: "org-1",
    organization_alias: "Test Organization 1",
    budget_id: "budget-1",
    metadata: {},
    models: ["gpt-3.5-turbo", "gpt-4"],
    spend: 100.5,
    model_spend: { "gpt-3.5-turbo": 50.25, "gpt-4": 50.25 },
    created_at: "2024-01-01T00:00:00Z",
    created_by: "user-1",
    updated_at: "2024-01-01T00:00:00Z",
    updated_by: "user-1",
    litellm_budget_table: null,
    teams: null,
    users: null,
    members: [
      { user_id: "user-1", user_role: "admin" },
      { user_id: "user-2", user_role: "member" },
    ],
  },
  {
    organization_id: "org-2",
    organization_alias: "Test Organization 2",
    budget_id: "budget-2",
    metadata: {},
    models: ["claude-3"],
    spend: 250.75,
    model_spend: { "claude-3": 250.75 },
    created_at: "2024-01-01T00:00:00Z",
    created_by: "user-3",
    updated_at: "2024-01-01T00:00:00Z",
    updated_by: "user-3",
    litellm_budget_table: null,
    teams: null,
    users: null,
    members: [{ user_id: "user-3", user_role: "admin" }],
  },
];

describe("useOrganizations", () => {
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
      userId: "test-user-id",
      userRole: "Admin",
      token: "test-token",
      userEmail: "test@example.com",
      premiumUser: false,
      disabledPersonalKeyCreation: null,
      showSSOBanner: false,
    });
  });

  const wrapper = ({ children }: { children: ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);

  it("should return organizations data when query is successful", async () => {
    // Mock successful API call
    (organizationListCall as any).mockResolvedValue(mockOrganizations);

    const { result } = renderHook(() => useOrganizations(), { wrapper });

    // Initially loading
    expect(result.current.isLoading).toBe(true);
    expect(result.current.data).toBeUndefined();

    // Wait for success
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual(mockOrganizations);
    expect(result.current.error).toBeNull();
    expect(organizationListCall).toHaveBeenCalledWith("test-access-token");
    expect(organizationListCall).toHaveBeenCalledTimes(1);
  });

  it("should handle error when organizationListCall fails", async () => {
    const errorMessage = "Failed to fetch organizations";
    const testError = new Error(errorMessage);

    // Mock failed API call
    (organizationListCall as any).mockRejectedValue(testError);

    const { result } = renderHook(() => useOrganizations(), { wrapper });

    // Initially loading
    expect(result.current.isLoading).toBe(true);

    // Wait for error
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(testError);
    expect(result.current.data).toBeUndefined();
    expect(organizationListCall).toHaveBeenCalledWith("test-access-token");
    expect(organizationListCall).toHaveBeenCalledTimes(1);
  });

  it("should not execute query when accessToken is missing", async () => {
    // Mock missing accessToken
    mockUseAuthorized.mockReturnValue({
      accessToken: null,
      userId: "test-user-id",
      userRole: "Admin",
      token: null,
      userEmail: "test@example.com",
      premiumUser: false,
      disabledPersonalKeyCreation: null,
      showSSOBanner: false,
    });

    const { result } = renderHook(() => useOrganizations(), { wrapper });

    // Query should not execute
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);

    // API should not be called
    expect(organizationListCall).not.toHaveBeenCalled();
  });

  it("should not execute query when userId is missing", async () => {
    // Mock missing userId
    mockUseAuthorized.mockReturnValue({
      accessToken: "test-access-token",
      userId: null,
      userRole: "Admin",
      token: "test-token",
      userEmail: "test@example.com",
      premiumUser: false,
      disabledPersonalKeyCreation: null,
      showSSOBanner: false,
    });

    const { result } = renderHook(() => useOrganizations(), { wrapper });

    // Query should not execute
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);

    // API should not be called
    expect(organizationListCall).not.toHaveBeenCalled();
  });

  it("should not execute query when userRole is missing", async () => {
    // Mock missing userRole
    mockUseAuthorized.mockReturnValue({
      accessToken: "test-access-token",
      userId: "test-user-id",
      userRole: null,
      token: "test-token",
      userEmail: "test@example.com",
      premiumUser: false,
      disabledPersonalKeyCreation: null,
      showSSOBanner: false,
    });

    const { result } = renderHook(() => useOrganizations(), { wrapper });

    // Query should not execute
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);

    // API should not be called
    expect(organizationListCall).not.toHaveBeenCalled();
  });

  it("should not execute query when all auth values are missing", async () => {
    // Mock all auth values missing
    mockUseAuthorized.mockReturnValue({
      accessToken: null,
      userId: null,
      userRole: null,
      token: null,
      userEmail: "test@example.com",
      premiumUser: false,
      disabledPersonalKeyCreation: null,
      showSSOBanner: false,
    });

    const { result } = renderHook(() => useOrganizations(), { wrapper });

    // Query should not execute
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);

    // API should not be called
    expect(organizationListCall).not.toHaveBeenCalled();
  });

  it("should execute query when all auth values are present", async () => {
    // Mock successful API call
    (organizationListCall as any).mockResolvedValue(mockOrganizations);

    // Ensure all auth values are present (already set in beforeEach)
    const { result } = renderHook(() => useOrganizations(), { wrapper });

    // Wait for query to execute
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(organizationListCall).toHaveBeenCalledWith("test-access-token");
    expect(organizationListCall).toHaveBeenCalledTimes(1);
  });

  it("should return empty array when API returns empty data", async () => {
    // Mock API returning empty array
    (organizationListCall as any).mockResolvedValue([]);

    const { result } = renderHook(() => useOrganizations(), { wrapper });

    // Wait for success
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual([]);
    expect(organizationListCall).toHaveBeenCalledWith("test-access-token");
  });

  it("should handle network timeout error", async () => {
    const timeoutError = new Error("Network timeout");

    // Mock network timeout
    (organizationListCall as any).mockRejectedValue(timeoutError);

    const { result } = renderHook(() => useOrganizations(), { wrapper });

    // Wait for error
    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(timeoutError);
    expect(result.current.data).toBeUndefined();
  });
});
