import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import React, { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useHealthReadinessDetails } from "./useHealthReadinessDetails";

vi.mock("@/components/networking", () => ({
  getProxyBaseUrl: () => "http://proxy.test",
  getGlobalLitellmHeaderName: () => "Authorization",
}));

const fetchMock = vi.fn();

describe("useHealthReadinessDetails", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    vi.clearAllMocks();
    fetchMock.mockResolvedValue({
      ok: true,
      statusText: "OK",
      json: async () => ({ status: "healthy", litellm_version: "9.9.9" }),
    });
    vi.stubGlobal("fetch", fetchMock);
  });

  const wrapper = ({ children }: { children: ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);

  it.each(["Admin", "Admin Viewer", "proxy_admin", "proxy_admin_viewer"])(
    "fetches /health/readiness/details for %s",
    async (role) => {
      const { result } = renderHook(() => useHealthReadinessDetails("sk-test", role), { wrapper });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(fetchMock).toHaveBeenCalledTimes(1);
      expect(fetchMock).toHaveBeenCalledWith("http://proxy.test/health/readiness/details", expect.any(Object));
      expect(result.current.data?.litellm_version).toBe("9.9.9");
    },
  );

  it.each(["Internal User", "Internal Viewer", "Org Admin", "App User", "Unknown Role", ""])(
    "never requests the admin-only readiness details for %s",
    async (role) => {
      const { result } = renderHook(() => useHealthReadinessDetails("sk-test", role), { wrapper });

      // The query is disabled, so give React Query a tick to prove nothing fires.
      await Promise.resolve();

      expect(fetchMock).not.toHaveBeenCalled();
      expect(result.current.isFetched).toBe(false);
      expect(result.current.data).toBeUndefined();
    },
  );

  it("reports missing details as absent rather than as an error, so the shell renders normally", async () => {
    const { result } = renderHook(() => useHealthReadinessDetails("sk-test", "Internal User"), { wrapper });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.isError).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it("stays inert in an unauthenticated shell, where there is no token and no role", async () => {
    const { result } = renderHook(() => useHealthReadinessDetails(null, null), { wrapper });

    await Promise.resolve();

    expect(fetchMock).not.toHaveBeenCalled();
    expect(result.current.isError).toBe(false);
    expect(result.current.data).toBeUndefined();
  });

  it("does not fetch for an admin whose token has not resolved yet", async () => {
    renderHook(() => useHealthReadinessDetails(null, "Admin"), { wrapper });

    await Promise.resolve();

    expect(fetchMock).not.toHaveBeenCalled();
  });
});
