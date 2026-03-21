/* @vitest-environment jsdom */
import React from "react";
import { renderHook, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useMCPServerHealth } from "./useMCPServerHealth";
import * as networking from "@/components/networking";

// Mock the networking module
vi.mock("@/components/networking", () => ({
  fetchMCPServerHealth: vi.fn(),
}));

// Mock useAuthorized hook
vi.mock("../useAuthorized", () => ({
  default: vi.fn(() => ({
    accessToken: "test-token-123",
  })),
}));

const createQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
      },
    },
  });

const wrapper = ({ children }: { children: React.ReactNode }) => {
  const queryClient = createQueryClient();
  return React.createElement(QueryClientProvider, { client: queryClient }, children);
};

describe("useMCPServerHealth", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should fetch health status for all servers", async () => {
    const mockHealthStatuses = [
      { server_id: "server-1", status: "healthy" },
      { server_id: "server-2", status: "unhealthy" },
    ];

    vi.mocked(networking.fetchMCPServerHealth).mockResolvedValue(mockHealthStatuses);

    const { result } = renderHook(() => useMCPServerHealth(), {
      wrapper,
    });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(networking.fetchMCPServerHealth).toHaveBeenCalledWith("test-token-123");
    expect(result.current.data).toEqual(mockHealthStatuses);
  });

  it("should handle errors when fetching health status", async () => {
    const mockError = new Error("Failed to fetch health status");
    vi.mocked(networking.fetchMCPServerHealth).mockRejectedValue(mockError);

    const { result } = renderHook(() => useMCPServerHealth(), {
      wrapper,
    });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(mockError);
  });

  it("should not fetch when accessToken is not available", async () => {
    // Mock useAuthorized to return no token
    const useAuthorizedModule = await import("../useAuthorized");
    vi.mocked(useAuthorizedModule.default).mockReturnValue({
      accessToken: null,
    } as any);

    const { result } = renderHook(() => useMCPServerHealth(), {
      wrapper,
    });

    // Should remain in idle state since query is not enabled
    expect(result.current.status).toBe("pending");
    expect(networking.fetchMCPServerHealth).not.toHaveBeenCalled();
  });

  it("should use a stable query key that does not include server IDs", () => {
    // Regression test: deleting a server used to pass a changing serverIds array into the
    // hook, which was embedded in the query key. React Query would see a new key and fire
    // a health check for every remaining server.
    //
    // The fix: the hook takes no serverIds parameter and uses a constant query key, so
    // deleting (or adding) a server never causes an extra health check request.
    //
    // We verify the contract here by confirming the hook accepts no arguments.
    // The stable-key behaviour is further exercised by mcp_servers.test.tsx.
    const hookLength = useMCPServerHealth.length;
    expect(hookLength).toBe(0);
  });
});
