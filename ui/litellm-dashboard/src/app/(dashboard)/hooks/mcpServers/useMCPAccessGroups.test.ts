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

const mockAccessToken = "test-token-456";
const mockAccessGroups = ["group-1", "group-2", "group-3"];

describe("useMCPAccessGroups", () => {
  beforeEach(async () => {
    vi.clearAllMocks();
    const useAuthorizedModule = await import("@/app/(dashboard)/hooks/useAuthorized");
    vi.mocked(useAuthorizedModule.default).mockReturnValue({
      accessToken: mockAccessToken,
    } as any);
  });

  it("should return hook result without errors", () => {
    vi.mocked(networking.fetchMCPAccessGroups).mockResolvedValue([]);

    const { result } = renderHook(() => useMCPAccessGroups(), { wrapper });

    expect(result.current).toBeDefined();
    expect(result.current).toHaveProperty("data");
    expect(result.current).toHaveProperty("isSuccess");
    expect(result.current).toHaveProperty("isError");
    expect(result.current).toHaveProperty("status");
  });

  it("should return MCP access groups when access token is present", async () => {
    vi.mocked(networking.fetchMCPAccessGroups).mockResolvedValue(mockAccessGroups);

    const { result } = renderHook(() => useMCPAccessGroups(), { wrapper });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(networking.fetchMCPAccessGroups).toHaveBeenCalledWith(mockAccessToken);
    expect(result.current.data).toEqual(mockAccessGroups);
  });

  it("should not fetch when access token is null", async () => {
    const useAuthorizedModule = await import("@/app/(dashboard)/hooks/useAuthorized");
    vi.mocked(useAuthorizedModule.default).mockReturnValue({
      accessToken: null,
    } as any);

    const { result } = renderHook(() => useMCPAccessGroups(), { wrapper });

    expect(result.current.isFetching).toBe(false);
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(networking.fetchMCPAccessGroups).not.toHaveBeenCalled();
  });

  it("should not fetch when access token is empty string", async () => {
    const useAuthorizedModule = await import("@/app/(dashboard)/hooks/useAuthorized");
    vi.mocked(useAuthorizedModule.default).mockReturnValue({
      accessToken: "",
    } as any);

    const { result } = renderHook(() => useMCPAccessGroups(), { wrapper });

    expect(result.current.isFetching).toBe(false);
    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
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
    expect(result.current.data).toBeUndefined();
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