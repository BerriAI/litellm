import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React, { ReactNode } from "react";
import { useAgents } from "./useAgents";
import { getAgentsList } from "@/components/networking";
import type { AgentsResponse, Agent } from "@/components/agents/types";

// Mock the networking function
vi.mock("@/components/networking", () => ({
  getAgentsList: vi.fn(),
}));

// Mock useAuthorized hook - we can override this in individual tests
const mockUseAuthorized = vi.fn();
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => mockUseAuthorized(),
}));

// Import actual roles instead of mocking them

// Mock data
const mockAgents: Agent[] = [
  {
    agent_id: "agent-1",
    agent_name: "Test Agent 1",
    litellm_params: {
      model: "gpt-3.5-turbo",
      api_key: "test-key-1",
    },
    agent_card_params: {
      description: "A test agent for unit testing",
    },
    created_at: "2024-01-01T00:00:00Z",
    updated_at: "2024-01-01T00:00:00Z",
    created_by: "user-1",
    updated_by: "user-1",
  },
  {
    agent_id: "agent-2",
    agent_name: "Test Agent 2",
    litellm_params: {
      model: "claude-3",
      api_key: "test-key-2",
    },
    agent_card_params: {
      description: "Another test agent",
    },
    created_at: "2024-01-01T00:00:00Z",
    updated_at: "2024-01-01T00:00:00Z",
    created_by: "user-2",
    updated_by: "user-2",
  },
];

const mockAgentsResponse: AgentsResponse = {
  agents: mockAgents,
};

describe("useAgents", () => {
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

  it("should return agents data when query is successful", async () => {
    // Mock successful API call
    (getAgentsList as any).mockResolvedValue(mockAgentsResponse);

    const { result } = renderHook(() => useAgents(), { wrapper });

    // Initially loading
    expect(result.current.isLoading).toBe(true);
    expect(result.current.data).toBeUndefined();

    // Wait for success
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual(mockAgentsResponse);
    expect(result.current.error).toBeNull();
    expect(getAgentsList).toHaveBeenCalledWith("test-access-token");
    expect(getAgentsList).toHaveBeenCalledTimes(1);
  });

  it("should handle error when getAgentsList fails", async () => {
    const errorMessage = "Failed to fetch agents";
    const testError = new Error(errorMessage);

    // Mock failed API call
    (getAgentsList as any).mockRejectedValue(testError);

    const { result } = renderHook(() => useAgents(), { wrapper });

    // Initially loading
    expect(result.current.isLoading).toBe(true);

    // Wait for error
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(testError);
    expect(result.current.data).toBeUndefined();
    expect(getAgentsList).toHaveBeenCalledWith("test-access-token");
    expect(getAgentsList).toHaveBeenCalledTimes(1);
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

    const { result } = renderHook(() => useAgents(), { wrapper });

    // Query should not execute
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);

    // API should not be called
    expect(getAgentsList).not.toHaveBeenCalled();
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

    const { result } = renderHook(() => useAgents(), { wrapper });

    // Query should not execute
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);

    // API should not be called
    expect(getAgentsList).not.toHaveBeenCalled();
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

    const { result } = renderHook(() => useAgents(), { wrapper });

    // Query should not execute
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);

    // API should not be called
    expect(getAgentsList).not.toHaveBeenCalled();
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

    const { result } = renderHook(() => useAgents(), { wrapper });

    // Query should not execute
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);

    // API should not be called
    expect(getAgentsList).not.toHaveBeenCalled();
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

    const { result } = renderHook(() => useAgents(), { wrapper });

    // Query should not execute
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);

    // API should not be called
    expect(getAgentsList).not.toHaveBeenCalled();
  });

  it("should execute query when accessToken is present and userRole is Admin", async () => {
    // Mock successful API call
    (getAgentsList as any).mockResolvedValue(mockAgentsResponse);

    // Ensure auth values are set (already done in beforeEach)
    const { result } = renderHook(() => useAgents(), { wrapper });

    // Wait for query to execute
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(getAgentsList).toHaveBeenCalledWith("test-access-token");
    expect(getAgentsList).toHaveBeenCalledTimes(1);
  });

  it("should execute query when accessToken is present and userRole is proxy_admin", async () => {
    // Mock successful API call
    (getAgentsList as any).mockResolvedValue(mockAgentsResponse);

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

    const { result } = renderHook(() => useAgents(), { wrapper });

    // Wait for query to execute
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(getAgentsList).toHaveBeenCalledWith("test-access-token");
    expect(getAgentsList).toHaveBeenCalledTimes(1);
  });

  it("should return empty agents array when API returns empty data", async () => {
    // Mock API returning empty agents array
    (getAgentsList as any).mockResolvedValue({ agents: [] });

    const { result } = renderHook(() => useAgents(), { wrapper });

    // Wait for success
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual({ agents: [] });
    expect(getAgentsList).toHaveBeenCalledWith("test-access-token");
  });

  it("should handle network timeout error", async () => {
    const timeoutError = new Error("Network timeout");

    // Mock network timeout
    (getAgentsList as any).mockRejectedValue(timeoutError);

    const { result } = renderHook(() => useAgents(), { wrapper });

    // Wait for error
    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(timeoutError);
    expect(result.current.data).toBeUndefined();
  });
});
