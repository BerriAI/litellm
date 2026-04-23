import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React, { ReactNode } from "react";
import { useEditSSOSettings, EditSSOSettingsParams, EditSSOSettingsResponse } from "./useEditSSOSettings";
import { updateSSOSettings } from "@/components/networking";

vi.mock("@/components/networking", () => ({
  updateSSOSettings: vi.fn(),
}));

const mockUseAuthorized = vi.fn();
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => mockUseAuthorized(),
}));

const mockUpdateResponse: EditSSOSettingsResponse = {
  message: "SSO settings updated successfully",
  google_client_id: "updated-google-client-id",
};

describe("useEditSSOSettings", () => {
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

  it("should render", () => {
    const { result } = renderHook(() => useEditSSOSettings(), { wrapper });

    expect(result.current).toBeDefined();
    expect(result.current.mutate).toBeDefined();
    expect(result.current.mutateAsync).toBeDefined();
  });

  it("should successfully update SSO settings", async () => {
    (updateSSOSettings as any).mockResolvedValue(mockUpdateResponse);

    const { result } = renderHook(() => useEditSSOSettings(), { wrapper });

    const params: EditSSOSettingsParams = {
      google_client_id: "new-google-client-id",
      google_client_secret: "new-google-client-secret",
    };

    result.current.mutateAsync(params);

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(updateSSOSettings).toHaveBeenCalledWith("test-access-token", params);
    expect(updateSSOSettings).toHaveBeenCalledTimes(1);
    expect(result.current.data).toEqual(mockUpdateResponse);
    expect(result.current.error).toBeNull();
  });

  it("should handle error when updateSSOSettings fails", async () => {
    const errorMessage = "Failed to update SSO settings";
    const testError = new Error(errorMessage);

    (updateSSOSettings as any).mockRejectedValue(testError);

    const { result } = renderHook(() => useEditSSOSettings(), { wrapper });

    const params: EditSSOSettingsParams = {
      google_client_id: "new-google-client-id",
    };

    result.current.mutateAsync(params).catch(() => {});

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(updateSSOSettings).toHaveBeenCalledWith("test-access-token", params);
    expect(result.current.error).toEqual(testError);
    expect(result.current.data).toBeUndefined();
  });

  it("should throw error when accessToken is missing", async () => {
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

    const { result } = renderHook(() => useEditSSOSettings(), { wrapper });

    const params: EditSSOSettingsParams = {
      google_client_id: "new-google-client-id",
    };

    await expect(result.current.mutateAsync(params)).rejects.toThrow("Access token is required");

    expect(updateSSOSettings).not.toHaveBeenCalled();
  });

  it("should update Microsoft SSO settings", async () => {
    (updateSSOSettings as any).mockResolvedValue(mockUpdateResponse);

    const { result } = renderHook(() => useEditSSOSettings(), { wrapper });

    const params: EditSSOSettingsParams = {
      microsoft_client_id: "new-microsoft-client-id",
      microsoft_client_secret: "new-microsoft-client-secret",
      microsoft_tenant: "new-tenant",
    };

    result.current.mutateAsync(params);

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(updateSSOSettings).toHaveBeenCalledWith("test-access-token", params);
  });

  it("should update generic SSO settings", async () => {
    (updateSSOSettings as any).mockResolvedValue(mockUpdateResponse);

    const { result } = renderHook(() => useEditSSOSettings(), { wrapper });

    const params: EditSSOSettingsParams = {
      generic_client_id: "new-generic-client-id",
      generic_client_secret: "new-generic-client-secret",
      generic_authorization_endpoint: "https://example.com/auth",
      generic_token_endpoint: "https://example.com/token",
      generic_userinfo_endpoint: "https://example.com/userinfo",
    };

    result.current.mutateAsync(params);

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(updateSSOSettings).toHaveBeenCalledWith("test-access-token", params);
  });

  it("should update role mappings", async () => {
    (updateSSOSettings as any).mockResolvedValue(mockUpdateResponse);

    const { result } = renderHook(() => useEditSSOSettings(), { wrapper });

    const params: EditSSOSettingsParams = {
      role_mappings: {
        provider: "google",
        group_claim: "groups",
        default_role: "internal_user",
        roles: {
          "admin-group": ["proxy_admin"],
        },
      },
    };

    result.current.mutateAsync(params);

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(updateSSOSettings).toHaveBeenCalledWith("test-access-token", params);
  });

  it("should update multiple settings at once", async () => {
    (updateSSOSettings as any).mockResolvedValue(mockUpdateResponse);

    const { result } = renderHook(() => useEditSSOSettings(), { wrapper });

    const params: EditSSOSettingsParams = {
      google_client_id: "new-google-client-id",
      microsoft_client_id: "new-microsoft-client-id",
      proxy_base_url: "https://new-proxy.example.com",
      user_email: "newuser@example.com",
      sso_provider: "google",
    };

    result.current.mutateAsync(params);

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(updateSSOSettings).toHaveBeenCalledWith("test-access-token", params);
  });

  it("should handle null values in params", async () => {
    (updateSSOSettings as any).mockResolvedValue(mockUpdateResponse);

    const { result } = renderHook(() => useEditSSOSettings(), { wrapper });

    const params: EditSSOSettingsParams = {
      google_client_id: null,
      google_client_secret: null,
    };

    result.current.mutateAsync(params);

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(updateSSOSettings).toHaveBeenCalledWith("test-access-token", params);
  });

  it("should set isPending to true during mutation", async () => {
    let resolvePromise: (value: EditSSOSettingsResponse) => void;
    const pendingPromise = new Promise<EditSSOSettingsResponse>((resolve) => {
      resolvePromise = resolve;
    });

    (updateSSOSettings as any).mockReturnValue(pendingPromise);

    const { result } = renderHook(() => useEditSSOSettings(), { wrapper });

    const params: EditSSOSettingsParams = {
      google_client_id: "new-google-client-id",
    };

    result.current.mutateAsync(params);

    await waitFor(() => {
      expect(result.current.isPending).toBe(true);
    });

    resolvePromise!(mockUpdateResponse);

    await waitFor(() => {
      expect(result.current.isPending).toBe(false);
    });
  });

  it("should handle network timeout error", async () => {
    const timeoutError = new Error("Network timeout");

    (updateSSOSettings as any).mockRejectedValue(timeoutError);

    const { result } = renderHook(() => useEditSSOSettings(), { wrapper });

    const params: EditSSOSettingsParams = {
      google_client_id: "new-google-client-id",
    };

    result.current.mutateAsync(params).catch(() => {});

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(timeoutError);
  });

  it("should reset error state on successful mutation after error", async () => {
    const errorMessage = "Failed to update";
    const testError = new Error(errorMessage);

    (updateSSOSettings as any).mockRejectedValueOnce(testError);

    const { result } = renderHook(() => useEditSSOSettings(), { wrapper });

    const params: EditSSOSettingsParams = {
      google_client_id: "new-google-client-id",
    };

    result.current.mutateAsync(params).catch(() => {});

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    (updateSSOSettings as any).mockResolvedValue(mockUpdateResponse);

    result.current.mutateAsync(params);

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
      expect(result.current.isError).toBe(false);
    });
  });
});
