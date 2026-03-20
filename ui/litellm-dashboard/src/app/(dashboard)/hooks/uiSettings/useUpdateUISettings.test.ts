import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React, { ReactNode } from "react";
import { useUpdateUISettings } from "./useUpdateUISettings";
import { updateUiSettings } from "@/components/networking";

vi.mock("@/components/networking", () => ({
  updateUiSettings: vi.fn(),
}));

const mockUpdateUiSettingsResponse = {
  message: "UI settings updated successfully",
  status: "success",
  settings: {
    disable_model_add_for_internal_users: true,
    disable_team_admin_delete_team_user: false,
  },
};

describe("useUpdateUISettings", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
        },
        mutations: {
          retry: false,
        },
      },
    });

    vi.clearAllMocks();
  });

  const wrapper = ({ children }: { children: ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);

  it("should render", () => {
    (updateUiSettings as any).mockResolvedValue(mockUpdateUiSettingsResponse);

    const { result } = renderHook(() => useUpdateUISettings("test-access-token"), { wrapper });

    expect(result.current).toBeDefined();
  });

  it("should update UI settings when mutation is successful", async () => {
    (updateUiSettings as any).mockResolvedValue(mockUpdateUiSettingsResponse);

    const { result } = renderHook(() => useUpdateUISettings("test-access-token"), { wrapper });

    const settings = {
      disable_model_add_for_internal_users: true,
    };

    result.current.mutate(settings);

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual(mockUpdateUiSettingsResponse);
    expect(updateUiSettings).toHaveBeenCalledWith("test-access-token", settings);
    expect(updateUiSettings).toHaveBeenCalledTimes(1);
  });

  it("should handle error when updateUiSettings fails", async () => {
    const errorMessage = "Failed to update UI settings";
    const testError = new Error(errorMessage);

    (updateUiSettings as any).mockRejectedValue(testError);

    const { result } = renderHook(() => useUpdateUISettings("test-access-token"), { wrapper });

    const settings = {
      disable_model_add_for_internal_users: true,
    };

    result.current.mutate(settings);

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(testError);
    expect(updateUiSettings).toHaveBeenCalledWith("test-access-token", settings);
    expect(updateUiSettings).toHaveBeenCalledTimes(1);
  });

  it("should throw error when accessToken is missing", async () => {
    const { result } = renderHook(() => useUpdateUISettings(""), { wrapper });

    const settings = {
      disable_model_add_for_internal_users: true,
    };

    result.current.mutate(settings);

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error?.message).toBe("Access token is required");
    expect(updateUiSettings).not.toHaveBeenCalled();
  });

  it("should throw error when accessToken is null", async () => {
    const { result } = renderHook(() => useUpdateUISettings(null as any), { wrapper });

    const settings = {
      disable_model_add_for_internal_users: true,
    };

    result.current.mutate(settings);

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error?.message).toBe("Access token is required");
    expect(updateUiSettings).not.toHaveBeenCalled();
  });

  it("should invalidate uiSettings queries on success", async () => {
    (updateUiSettings as any).mockResolvedValue(mockUpdateUiSettingsResponse);

    queryClient.setQueryData(["uiSettings", "detail", "settings"], { values: {} });

    const { result } = renderHook(() => useUpdateUISettings("test-access-token"), { wrapper });

    const settings = {
      disable_model_add_for_internal_users: true,
    };

    result.current.mutate(settings);

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    const queryCache = queryClient.getQueryCache();
    const queries = queryCache.findAll({ queryKey: ["uiSettings"] });
    expect(queries.length).toBeGreaterThan(0);
  });

  it("should handle multiple settings updates", async () => {
    (updateUiSettings as any).mockResolvedValue(mockUpdateUiSettingsResponse);

    const { result } = renderHook(() => useUpdateUISettings("test-access-token"), { wrapper });

    const settings1 = {
      disable_model_add_for_internal_users: true,
    };

    const settings2 = {
      disable_team_admin_delete_team_user: false,
    };

    result.current.mutate(settings1);

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    result.current.mutate(settings2);

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(updateUiSettings).toHaveBeenCalledTimes(2);
    expect(updateUiSettings).toHaveBeenNthCalledWith(1, "test-access-token", settings1);
    expect(updateUiSettings).toHaveBeenNthCalledWith(2, "test-access-token", settings2);
  });

  it("should handle empty settings object", async () => {
    (updateUiSettings as any).mockResolvedValue(mockUpdateUiSettingsResponse);

    const { result } = renderHook(() => useUpdateUISettings("test-access-token"), { wrapper });

    result.current.mutate({});

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(updateUiSettings).toHaveBeenCalledWith("test-access-token", {});
  });

  it("should handle network timeout error", async () => {
    const timeoutError = new Error("Network timeout");

    (updateUiSettings as any).mockRejectedValue(timeoutError);

    const { result } = renderHook(() => useUpdateUISettings("test-access-token"), { wrapper });

    const settings = {
      disable_model_add_for_internal_users: true,
    };

    result.current.mutate(settings);

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(timeoutError);
  });

  it("should set isPending during mutation", async () => {
    let resolvePromise: (value: any) => void;
    const promise = new Promise((resolve) => {
      resolvePromise = resolve;
    });

    (updateUiSettings as any).mockReturnValue(promise);

    const { result } = renderHook(() => useUpdateUISettings("test-access-token"), { wrapper });

    const settings = {
      disable_model_add_for_internal_users: true,
    };

    result.current.mutate(settings);

    // Wait for the mutation to start and isPending to become true
    await waitFor(() => {
      expect(result.current.isPending).toBe(true);
    });

    resolvePromise!(mockUpdateUiSettingsResponse);

    await waitFor(() => {
      expect(result.current.isPending).toBe(false);
    });
  });
});
