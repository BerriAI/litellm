import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React, { ReactNode } from "react";
import { useStoreModelInDB } from "./useStoreModelInDB";

vi.mock("@/components/networking", () => ({
  getProxyBaseUrl: vi.fn(() => ""),
  getGlobalLitellmHeaderName: vi.fn(() => "Authorization"),
}));

describe("useStoreModelInDB", () => {
  let queryClient: QueryClient;
  let fetchSpy: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        mutations: { retry: false },
      },
    });

    fetchSpy = vi.fn();
    global.fetch = fetchSpy;
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  const wrapper = ({ children }: { children: ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);

  it("should send correct request body to /config/field/update", async () => {
    fetchSpy.mockResolvedValue({
      ok: true,
      json: async () => ({ message: "Success" }),
    });

    const { result } = renderHook(() => useStoreModelInDB(), { wrapper });

    result.current.mutate({ store_model_in_db: true });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(fetchSpy).toHaveBeenCalledWith(
      "/config/field/update",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          field_name: "store_model_in_db",
          field_value: true,
          config_type: "general_settings",
        }),
      })
    );
  });

  it("should handle setting store_model_in_db to false", async () => {
    fetchSpy.mockResolvedValue({
      ok: true,
      json: async () => ({ message: "Success" }),
    });

    const { result } = renderHook(() => useStoreModelInDB(), { wrapper });

    result.current.mutate({ store_model_in_db: false });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(fetchSpy).toHaveBeenCalledWith(
      "/config/field/update",
      expect.objectContaining({
        body: JSON.stringify({
          field_name: "store_model_in_db",
          field_value: false,
          config_type: "general_settings",
        }),
      })
    );
  });

  it("should throw error when access token is missing", async () => {
    vi.spyOn(
      await import("../useAuthorized"),
      "default"
    ).mockReturnValue({
      accessToken: null,
      userRole: null,
      userId: null,
      token: null,
      userEmail: null,
      premiumUser: false,
      disabledPersonalKeyCreation: null,
      showSSOBanner: false,
    } as any);

    const { result } = renderHook(() => useStoreModelInDB(), { wrapper });

    result.current.mutate({ store_model_in_db: true });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error?.message).toBe("Access token is required");
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("should handle API error response", async () => {
    fetchSpy.mockResolvedValue({
      ok: false,
      json: async () => ({ detail: "Unauthorized" }),
    });

    const { result } = renderHook(() => useStoreModelInDB(), { wrapper });

    result.current.mutate({ store_model_in_db: true });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error?.message).toBe("Unauthorized");
  });

  it("should use fallback error message when API returns empty error", async () => {
    fetchSpy.mockResolvedValue({
      ok: false,
      json: async () => ({}),
    });

    const { result } = renderHook(() => useStoreModelInDB(), { wrapper });

    result.current.mutate({ store_model_in_db: true });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error?.message).toBe("Failed to update model storage settings");
  });
});
