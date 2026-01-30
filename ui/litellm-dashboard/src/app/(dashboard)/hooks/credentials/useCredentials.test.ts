import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React, { ReactNode } from "react";
import { useCredentials } from "./useCredentials";
import { credentialListCall, CredentialsResponse, CredentialItem } from "@/components/networking";

// Mock the networking function
vi.mock("@/components/networking", () => ({
  credentialListCall: vi.fn(),
}));

// Mock useAuthorized hook - we can override this in individual tests
const mockUseAuthorized = vi.fn();
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => mockUseAuthorized(),
}));

// Mock data
const mockCredentialItems: CredentialItem[] = [
  {
    credential_name: "openai-api-key",
    credential_values: { api_key: "sk-test123" },
    credential_info: {
      custom_llm_provider: "openai",
      description: "OpenAI API Key for GPT models",
      required: true,
    },
  },
  {
    credential_name: "anthropic-api-key",
    credential_values: { api_key: "sk-ant-test456" },
    credential_info: {
      custom_llm_provider: "anthropic",
      description: "Anthropic API Key for Claude models",
      required: true,
    },
  },
];

const mockCredentialsResponse: CredentialsResponse = {
  credentials: mockCredentialItems,
};

describe("useCredentials", () => {
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

  it("should return credentials data when query is successful", async () => {
    // Mock successful API call
    (credentialListCall as any).mockResolvedValue(mockCredentialsResponse);

    const { result } = renderHook(() => useCredentials(), { wrapper });

    // Initially loading
    expect(result.current.isLoading).toBe(true);
    expect(result.current.data).toBeUndefined();

    // Wait for success
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual(mockCredentialsResponse);
    expect(result.current.error).toBeNull();
    expect(credentialListCall).toHaveBeenCalledWith("test-access-token");
    expect(credentialListCall).toHaveBeenCalledTimes(1);
  });

  it("should handle error when credentialListCall fails", async () => {
    const errorMessage = "Failed to fetch credentials";
    const testError = new Error(errorMessage);

    // Mock failed API call
    (credentialListCall as any).mockRejectedValue(testError);

    const { result } = renderHook(() => useCredentials(), { wrapper });

    // Initially loading
    expect(result.current.isLoading).toBe(true);

    // Wait for error
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(testError);
    expect(result.current.data).toBeUndefined();
    expect(credentialListCall).toHaveBeenCalledWith("test-access-token");
    expect(credentialListCall).toHaveBeenCalledTimes(1);
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

    const { result } = renderHook(() => useCredentials(), { wrapper });

    // Query should not execute
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);

    // API should not be called
    expect(credentialListCall).not.toHaveBeenCalled();
  });

  it("should return empty credentials array when API returns empty data", async () => {
    // Mock API returning empty credentials array
    (credentialListCall as any).mockResolvedValue({ credentials: [] });

    const { result } = renderHook(() => useCredentials(), { wrapper });

    // Wait for success
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual({ credentials: [] });
    expect(credentialListCall).toHaveBeenCalledWith("test-access-token");
  });

  it("should handle network timeout error", async () => {
    const timeoutError = new Error("Network timeout");

    // Mock network timeout
    (credentialListCall as any).mockRejectedValue(timeoutError);

    const { result } = renderHook(() => useCredentials(), { wrapper });

    // Wait for error
    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(timeoutError);
    expect(result.current.data).toBeUndefined();
  });

  it("should execute query when accessToken is present", async () => {
    // Mock successful API call
    (credentialListCall as any).mockResolvedValue(mockCredentialsResponse);

    // Ensure auth values are set (already done in beforeEach)
    const { result } = renderHook(() => useCredentials(), { wrapper });

    // Wait for query to execute
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(credentialListCall).toHaveBeenCalledWith("test-access-token");
    expect(credentialListCall).toHaveBeenCalledTimes(1);
  });
});
