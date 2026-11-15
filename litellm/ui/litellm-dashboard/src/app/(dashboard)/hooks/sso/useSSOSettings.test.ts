import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React, { ReactNode } from "react";
import { useSSOSettings, SSOSettingsResponse } from "./useSSOSettings";
import { getSSOSettings } from "@/components/networking";

vi.mock("@/components/networking", () => ({
  getSSOSettings: vi.fn(),
}));

const mockUseAuthorized = vi.fn();
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => mockUseAuthorized(),
}));

const mockSSOSettingsResponse: SSOSettingsResponse = {
  values: {
    google_client_id: "test-google-client-id",
    google_client_secret: "test-google-client-secret",
    microsoft_client_id: "test-microsoft-client-id",
    microsoft_client_secret: "test-microsoft-client-secret",
    microsoft_tenant: "test-tenant",
    generic_client_id: "test-generic-client-id",
    generic_client_secret: "test-generic-client-secret",
    generic_authorization_endpoint: "https://example.com/auth",
    generic_token_endpoint: "https://example.com/token",
    generic_userinfo_endpoint: "https://example.com/userinfo",
    proxy_base_url: "https://proxy.example.com",
    user_email: "test@example.com",
    ui_access_mode: "proxy_admin",
    role_mappings: {
      provider: "google",
      group_claim: "groups",
      default_role: "internal_user",
      roles: {
        "admin-group": ["proxy_admin"],
        "viewer-group": ["internal_user_viewer"],
      },
    },
    team_mappings: {
      team_ids_jwt_field: "team_ids",
    },
  },
  field_schema: {
    description: "SSO Settings Schema",
    properties: {
      google_client_id: {
        description: "Google OAuth Client ID",
        type: "string",
      },
      microsoft_client_id: {
        description: "Microsoft OAuth Client ID",
        type: "string",
      },
    },
  },
};

describe("useSSOSettings", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: {
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
    (getSSOSettings as any).mockResolvedValue(mockSSOSettingsResponse);

    const { result } = renderHook(() => useSSOSettings(), { wrapper });

    expect(result.current).toBeDefined();
  });

  it("should return SSO settings data when query is successful", async () => {
    (getSSOSettings as any).mockResolvedValue(mockSSOSettingsResponse);

    const { result } = renderHook(() => useSSOSettings(), { wrapper });

    expect(result.current.isLoading).toBe(true);
    expect(result.current.data).toBeUndefined();

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual(mockSSOSettingsResponse);
    expect(result.current.error).toBeNull();
    expect(getSSOSettings).toHaveBeenCalledWith("test-access-token");
    expect(getSSOSettings).toHaveBeenCalledTimes(1);
  });

  it("should handle error when getSSOSettings fails", async () => {
    const errorMessage = "Failed to fetch SSO settings";
    const testError = new Error(errorMessage);

    (getSSOSettings as any).mockRejectedValue(testError);

    const { result } = renderHook(() => useSSOSettings(), { wrapper });

    expect(result.current.isLoading).toBe(true);

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(testError);
    expect(result.current.data).toBeUndefined();
    expect(getSSOSettings).toHaveBeenCalledWith("test-access-token");
    expect(getSSOSettings).toHaveBeenCalledTimes(1);
  });

  it("should not execute query when accessToken is missing", async () => {
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

    const { result } = renderHook(() => useSSOSettings(), { wrapper });

    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);

    expect(getSSOSettings).not.toHaveBeenCalled();
  });

  it("should not execute query when userId is missing", async () => {
    mockUseAuthorized.mockReturnValue({
      accessToken: "test-access-token",
      userId: null,
      userRole: "Admin",
      token: "test-token",
      userEmail: "test@example.com",
      premiumUser: false,
      disabledPersonalKeyCreation: null,
      showSSOBanner: false,
    });

    const { result } = renderHook(() => useSSOSettings(), { wrapper });

    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);

    expect(getSSOSettings).not.toHaveBeenCalled();
  });

  it("should not execute query when userRole is missing", async () => {
    mockUseAuthorized.mockReturnValue({
      accessToken: "test-access-token",
      userId: "test-user-id",
      userRole: null,
      token: "test-token",
      userEmail: "test@example.com",
      premiumUser: false,
      disabledPersonalKeyCreation: null,
      showSSOBanner: false,
    });

    const { result } = renderHook(() => useSSOSettings(), { wrapper });

    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);

    expect(getSSOSettings).not.toHaveBeenCalled();
  });

  it("should not execute query when all auth values are missing", async () => {
    mockUseAuthorized.mockReturnValue({
      accessToken: null,
      userId: null,
      userRole: null,
      token: null,
      userEmail: "test@example.com",
      premiumUser: false,
      disabledPersonalKeyCreation: null,
      showSSOBanner: false,
    });

    const { result } = renderHook(() => useSSOSettings(), { wrapper });

    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);

    expect(getSSOSettings).not.toHaveBeenCalled();
  });

  it("should execute query when all auth values are present", async () => {
    (getSSOSettings as any).mockResolvedValue(mockSSOSettingsResponse);

    const { result } = renderHook(() => useSSOSettings(), { wrapper });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(getSSOSettings).toHaveBeenCalledWith("test-access-token");
    expect(getSSOSettings).toHaveBeenCalledTimes(1);
  });

  it("should return empty values when API returns minimal data", async () => {
    const minimalResponse: SSOSettingsResponse = {
      values: {
        google_client_id: null,
        google_client_secret: null,
        microsoft_client_id: null,
        microsoft_client_secret: null,
        microsoft_tenant: null,
        generic_client_id: null,
        generic_client_secret: null,
        generic_authorization_endpoint: null,
        generic_token_endpoint: null,
        generic_userinfo_endpoint: null,
        proxy_base_url: null,
        user_email: null,
        ui_access_mode: null,
        role_mappings: {
          provider: "",
          group_claim: "",
          default_role: "internal_user",
          roles: {},
        },
        team_mappings: {
          team_ids_jwt_field: "",
        },
      },
      field_schema: {
        description: "",
        properties: {},
      },
    };

    (getSSOSettings as any).mockResolvedValue(minimalResponse);

    const { result } = renderHook(() => useSSOSettings(), { wrapper });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual(minimalResponse);
    expect(getSSOSettings).toHaveBeenCalledWith("test-access-token");
  });

  it("should handle network timeout error", async () => {
    const timeoutError = new Error("Network timeout");

    (getSSOSettings as any).mockRejectedValue(timeoutError);

    const { result } = renderHook(() => useSSOSettings(), { wrapper });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(timeoutError);
    expect(result.current.data).toBeUndefined();
  });

  it("should use correct query key", async () => {
    (getSSOSettings as any).mockResolvedValue(mockSSOSettingsResponse);

    const { result } = renderHook(() => useSSOSettings(), { wrapper });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    const queryCache = queryClient.getQueryCache();
    const queries = queryCache.findAll();
    const ssoQuery = queries.find((q) => q.queryKey[0] === "sso");

    expect(ssoQuery).toBeDefined();
    expect(ssoQuery?.queryKey).toEqual(["sso", "detail", "settings"]);
  });
});
