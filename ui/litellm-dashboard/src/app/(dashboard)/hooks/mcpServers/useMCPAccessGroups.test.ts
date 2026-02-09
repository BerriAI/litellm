/* @vitest-environment jsdom */
import React from "react";
import { renderHook, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useMCPAccessGroups } from "./useMCPAccessGroups";
import * as networking from "@/components/networking";

vi.mock("@/components/networking", () => ({
  fetchMCPAccessGroups: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: vi.fn(() => ({
    accessToken: "test-token-456",
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

const mockAccessGroups = ["group-1", "group-2", "group-3"];

describe("useMCPAccessGroups", () => {
  beforeEach(async () => {
    vi.clearAllMocks();
    const useAuthorizedModule = await import("@/app/(dashboard)/hooks/useAuthorized");
    vi.mocked(useAuthorizedModule.default).mockReturnValue({
      accessToken: "test-token-456",
    } as any);
  });

  it("should return MCP access groups when access token is present", async () => {
    vi.mocked(networking.fetchMCPAccessGroups).mockResolvedValue(mockAccessGroups);

    const { result } = renderHook(() => useMCPAccessGroups(), { wrapper });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(networking.fetchMCPAccessGroups).toHaveBeenCalledWith("test-token-456");
    expect(result.current.data).toEqual(mockAccessGroups);
  });

  it("should not fetch when access token is not available", async () => {
    const useAuthorizedModule = await import("@/app/(dashboard)/hooks/useAuthorized");
    vi.mocked(useAuthorizedModule.default).mockReturnValue({
      accessToken: null,
    } as any);

    const { result } = renderHook(() => useMCPAccessGroups(), { wrapper });

    expect(result.current.status).toBe("pending");
    expect(networking.fetchMCPAccessGroups).not.toHaveBeenCalled();
  });

  it("should expose error state when fetch fails", async () => {
    const mockError = new Error("Failed to fetch MCP access groups");
    vi.mocked(networking.fetchMCPAccessGroups).mockRejectedValue(mockError);

    const { result } = renderHook(() => useMCPAccessGroups(), { wrapper });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(mockError);
  });

  it("should return empty array when API returns no groups", async () => {
    vi.mocked(networking.fetchMCPAccessGroups).mockResolvedValue([]);

    const { result } = renderHook(() => useMCPAccessGroups(), { wrapper });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual([]);
  });
});