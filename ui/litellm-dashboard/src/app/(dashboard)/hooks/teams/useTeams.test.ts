import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React, { ReactNode } from "react";
import { useTeams } from "./useTeams";
import { fetchTeams } from "@/app/(dashboard)/networking";
import type { Team } from "@/components/key_team_helpers/key_list";

// Mock the networking function
vi.mock("@/app/(dashboard)/networking", () => ({
  fetchTeams: vi.fn(),
}));

// Mock useAuthorized hook - we can override this in individual tests
const mockUseAuthorized = vi.fn();
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => mockUseAuthorized(),
}));

// Mock data
const mockTeams: Team[] = [
  {
    team_id: "team-1",
    team_alias: "Test Team 1",
    models: ["gpt-3.5-turbo", "claude-3"],
    max_budget: 100.0,
    budget_duration: "monthly",
    tpm_limit: 1000,
    rpm_limit: 100,
    organization_id: "org-1",
    created_at: "2024-01-01T00:00:00Z",
    keys: [],
    members_with_roles: [],
  },
  {
    team_id: "team-2",
    team_alias: "Test Team 2",
    models: ["gpt-4"],
    max_budget: 200.0,
    budget_duration: "monthly",
    tpm_limit: 2000,
    rpm_limit: 200,
    organization_id: "org-1",
    created_at: "2024-01-02T00:00:00Z",
    keys: [],
    members_with_roles: [],
  },
];

describe("useTeams", () => {
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

  it("should return teams data when query is successful", async () => {
    // Mock successful API call
    (fetchTeams as any).mockResolvedValue(mockTeams);

    const { result } = renderHook(() => useTeams(), { wrapper });

    // Initially loading
    expect(result.current.isLoading).toBe(true);
    expect(result.current.data).toBeUndefined();

    // Wait for success
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual(mockTeams);
    expect(result.current.error).toBeNull();
    expect(fetchTeams).toHaveBeenCalledWith("test-access-token", "test-user-id", "Admin", null);
    expect(fetchTeams).toHaveBeenCalledTimes(1);
  });

  it("should handle error when fetchTeams fails", async () => {
    const errorMessage = "Failed to fetch teams";
    const testError = new Error(errorMessage);

    // Mock failed API call
    (fetchTeams as any).mockRejectedValue(testError);

    const { result } = renderHook(() => useTeams(), { wrapper });

    // Initially loading
    expect(result.current.isLoading).toBe(true);

    // Wait for error
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(testError);
    expect(result.current.data).toBeUndefined();
    expect(fetchTeams).toHaveBeenCalledWith("test-access-token", "test-user-id", "Admin", null);
    expect(fetchTeams).toHaveBeenCalledTimes(1);
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

    const { result } = renderHook(() => useTeams(), { wrapper });

    // Query should not execute
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);

    // API should not be called
    expect(fetchTeams).not.toHaveBeenCalled();
  });

  it("should not execute query when accessToken is empty string", async () => {
    // Mock empty string accessToken
    mockUseAuthorized.mockReturnValue({
      accessToken: "",
      userId: "test-user-id",
      userRole: "Admin",
      token: "",
      userEmail: "test@example.com",
      premiumUser: false,
      disabledPersonalKeyCreation: null,
      showSSOBanner: false,
    });

    const { result } = renderHook(() => useTeams(), { wrapper });

    // Query should not execute
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);

    // API should not be called
    expect(fetchTeams).not.toHaveBeenCalled();
  });

  it("should execute query when accessToken is present", async () => {
    // Mock successful API call
    (fetchTeams as any).mockResolvedValue(mockTeams);

    // Ensure auth values are set (already done in beforeEach)
    const { result } = renderHook(() => useTeams(), { wrapper });

    // Wait for query to execute
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(fetchTeams).toHaveBeenCalledWith("test-access-token", "test-user-id", "Admin", null);
    expect(fetchTeams).toHaveBeenCalledTimes(1);
  });

  it("should return empty teams array when API returns empty data", async () => {
    // Mock API returning empty teams array
    (fetchTeams as any).mockResolvedValue([]);

    const { result } = renderHook(() => useTeams(), { wrapper });

    // Wait for success
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual([]);
    expect(fetchTeams).toHaveBeenCalledWith("test-access-token", "test-user-id", "Admin", null);
  });

  it("should handle network timeout error", async () => {
    const timeoutError = new Error("Network timeout");

    // Mock network timeout
    (fetchTeams as any).mockRejectedValue(timeoutError);

    const { result } = renderHook(() => useTeams(), { wrapper });

    // Wait for error
    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(timeoutError);
    expect(result.current.data).toBeUndefined();
  });

  it("should pass userId and userRole to fetchTeams", async () => {
    // Mock successful API call
    (fetchTeams as any).mockResolvedValue(mockTeams);

    // Mock specific userId and userRole
    mockUseAuthorized.mockReturnValue({
      accessToken: "test-access-token",
      userId: "custom-user-id",
      userRole: "member",
      token: "test-token",
      userEmail: "test@example.com",
      premiumUser: false,
      disabledPersonalKeyCreation: null,
      showSSOBanner: false,
    });

    const { result } = renderHook(() => useTeams(), { wrapper });

    // Wait for query to execute
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(fetchTeams).toHaveBeenCalledWith("test-access-token", "custom-user-id", "member", null);
  });

  it("should handle null userId", async () => {
    // Mock successful API call
    (fetchTeams as any).mockResolvedValue(mockTeams);

    // Mock null userId
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

    const { result } = renderHook(() => useTeams(), { wrapper });

    // Wait for query to execute
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(fetchTeams).toHaveBeenCalledWith("test-access-token", null, "Admin", null);
  });
});
