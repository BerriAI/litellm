/* @vitest-environment jsdom */
import React from "react";
import { renderHook, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useMCPServers } from "./useMCPServers";
import * as networking from "@/components/networking";

vi.mock("@/components/networking", () => ({
  fetchMCPServers: vi.fn(),
}));

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

const mockServers = [
  {
    server_id: "server-1",
    server_name: "Server One",
    url: "http://localhost:4000",
    created_at: "2025-01-01T00:00:00Z",
    created_by: "user-1",
    updated_at: "2025-01-01T00:00:00Z",
    updated_by: "user-1",
  },
];

describe("useMCPServers", () => {
  beforeEach(async () => {
    vi.clearAllMocks();
    const useAuthorizedModule = await import("../useAuthorized");
    vi.mocked(useAuthorizedModule.default).mockReturnValue({
      accessToken: "test-token-123",
    } as any);
  });

  it("should return MCP servers when access token is present", async () => {
    vi.mocked(networking.fetchMCPServers).mockResolvedValue(mockServers);

    const { result } = renderHook(() => useMCPServers(), { wrapper });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(networking.fetchMCPServers).toHaveBeenCalledWith("test-token-123");
    expect(result.current.data).toEqual(mockServers);
  });

  it("should not fetch when access token is not available", async () => {
    const useAuthorizedModule = await import("../useAuthorized");
    vi.mocked(useAuthorizedModule.default).mockReturnValue({
      accessToken: null,
    } as any);

    const { result } = renderHook(() => useMCPServers(), { wrapper });

    expect(result.current.status).toBe("pending");
    expect(networking.fetchMCPServers).not.toHaveBeenCalled();
  });

  it("should expose error state when fetch fails", async () => {
    const mockError = new Error("Failed to fetch MCP servers");
    vi.mocked(networking.fetchMCPServers).mockRejectedValue(mockError);

    const { result } = renderHook(() => useMCPServers(), { wrapper });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(mockError);
  });

  it("should return empty array when API returns empty list", async () => {
    vi.mocked(networking.fetchMCPServers).mockResolvedValue([]);

    const { result } = renderHook(() => useMCPServers(), { wrapper });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual([]);
  });
});