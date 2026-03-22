import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React, { ReactNode } from "react";
import { useGuardrails } from "./useGuardrails";
import { getGuardrailsList } from "@/components/networking";

// Mock the networking function
vi.mock("@/components/networking", () => ({
  getGuardrailsList: vi.fn(),
}));

// Mock useAuthorized hook - we can override this in individual tests
const mockUseAuthorized = vi.fn();
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => mockUseAuthorized(),
}));

// Mock data
const mockGuardrailsResponse = {
  guardrails: [
    { guardrail_name: "content-safety" },
    { guardrail_name: "toxicity-filter" },
    { guardrail_name: "pii-detection" },
  ],
};

const expectedGuardrailNames = ["content-safety", "toxicity-filter", "pii-detection"];

describe("useGuardrails", () => {
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

  it("should return guardrail names when query is successful", async () => {
    // Mock successful API call
    (getGuardrailsList as any).mockResolvedValue(mockGuardrailsResponse);

    const { result } = renderHook(() => useGuardrails(), { wrapper });

    // Initially loading
    expect(result.current.isLoading).toBe(true);
    expect(result.current.data).toBeUndefined();

    // Wait for success
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual(expectedGuardrailNames);
    expect(result.current.error).toBeNull();
    expect(getGuardrailsList).toHaveBeenCalledWith("test-access-token");
    expect(getGuardrailsList).toHaveBeenCalledTimes(1);
  });

  it("should handle error when getGuardrailsList fails", async () => {
    const errorMessage = "Failed to fetch guardrails";
    const testError = new Error(errorMessage);

    // Mock failed API call
    (getGuardrailsList as any).mockRejectedValue(testError);

    const { result } = renderHook(() => useGuardrails(), { wrapper });

    // Initially loading
    expect(result.current.isLoading).toBe(true);

    // Wait for error
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(testError);
    expect(result.current.data).toBeUndefined();
    expect(getGuardrailsList).toHaveBeenCalledWith("test-access-token");
    expect(getGuardrailsList).toHaveBeenCalledTimes(1);
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

    const { result } = renderHook(() => useGuardrails(), { wrapper });

    // Query should not execute
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);

    // API should not be called
    expect(getGuardrailsList).not.toHaveBeenCalled();
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

    const { result } = renderHook(() => useGuardrails(), { wrapper });

    // Query should not execute
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);

    // API should not be called
    expect(getGuardrailsList).not.toHaveBeenCalled();
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

    const { result } = renderHook(() => useGuardrails(), { wrapper });

    // Query should not execute
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);

    // API should not be called
    expect(getGuardrailsList).not.toHaveBeenCalled();
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

    const { result } = renderHook(() => useGuardrails(), { wrapper });

    // Query should not execute
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);

    // API should not be called
    expect(getGuardrailsList).not.toHaveBeenCalled();
  });

  it("should execute query when all auth values are present", async () => {
    // Mock successful API call
    (getGuardrailsList as any).mockResolvedValue(mockGuardrailsResponse);

    // Ensure all auth values are present (already set in beforeEach)
    const { result } = renderHook(() => useGuardrails(), { wrapper });

    // Wait for query to execute
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(getGuardrailsList).toHaveBeenCalledWith("test-access-token");
    expect(getGuardrailsList).toHaveBeenCalledTimes(1);
  });

  it("should return empty array when API returns empty guardrails", async () => {
    // Mock API returning empty guardrails array
    (getGuardrailsList as any).mockResolvedValue({ guardrails: [] });

    const { result } = renderHook(() => useGuardrails(), { wrapper });

    // Wait for success
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual([]);
    expect(getGuardrailsList).toHaveBeenCalledWith("test-access-token");
  });

  it("should handle network timeout error", async () => {
    const timeoutError = new Error("Network timeout");

    // Mock network timeout
    (getGuardrailsList as any).mockRejectedValue(timeoutError);

    const { result } = renderHook(() => useGuardrails(), { wrapper });

    // Wait for error
    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(timeoutError);
    expect(result.current.data).toBeUndefined();
  });

  it("should correctly transform guardrail objects to names array", async () => {
    const customGuardrailsResponse = {
      guardrails: [{ guardrail_name: "custom-guardrail-1" }, { guardrail_name: "custom-guardrail-2" }],
    };
    const expectedNames = ["custom-guardrail-1", "custom-guardrail-2"];

    // Mock API call with custom data
    (getGuardrailsList as any).mockResolvedValue(customGuardrailsResponse);

    const { result } = renderHook(() => useGuardrails(), { wrapper });

    // Wait for success
    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual(expectedNames);
    expect(result.current.data).toHaveLength(2);
    expect(result.current.data).toContain("custom-guardrail-1");
    expect(result.current.data).toContain("custom-guardrail-2");
  });
});
