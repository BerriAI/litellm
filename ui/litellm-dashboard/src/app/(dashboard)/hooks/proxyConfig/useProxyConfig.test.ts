import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React, { ReactNode } from "react";
import {
  useProxyConfig,
  useDeleteProxyConfigField,
  getProxyConfigCall,
  deleteProxyConfigFieldCall,
  ConfigType,
  GeneralSettingsFieldName,
  type ProxyConfigResponse,
  type DeleteProxyConfigFieldRequest,
  type DeleteProxyConfigFieldResponse,
} from "./useProxyConfig";

const {
  mockProxyBaseUrl,
  mockAccessToken,
  mockHeaderName,
  mockProxyConfigResponse,
  mockDeleteResponse,
  mockUseAuthorized,
  mockGetGlobalLitellmHeaderName,
  mockDeriveErrorMessage,
  mockHandleError,
} = vi.hoisted(() => {
  const mockProxyBaseUrl = "https://proxy.example.com";
  const mockAccessToken = "test-access-token";
  const mockHeaderName = "X-LiteLLM-API-Key";

  const mockProxyConfigResponse: ProxyConfigResponse = [
    {
      field_name: "maximum_spend_logs_retention_period",
      field_type: "int",
      field_description: "Maximum retention period for spend logs",
      field_value: 30,
      stored_in_db: true,
      field_default_value: 7,
      premium_field: false,
      nested_fields: null,
    },
    {
      field_name: "another_field",
      field_type: "string",
      field_description: "Another config field",
      field_value: "test-value",
      stored_in_db: false,
      field_default_value: "default-value",
      premium_field: true,
      nested_fields: [
        {
          field_name: "nested_field",
          field_type: "string",
          field_description: "Nested field description",
          field_default_value: "nested-default",
          stored_in_db: true,
        },
      ],
    },
  ];

  const mockDeleteResponse: DeleteProxyConfigFieldResponse = {
    message: "Field deleted successfully",
  };

  const mockUseAuthorized = vi.fn();
  const mockGetGlobalLitellmHeaderName = vi.fn(() => mockHeaderName);
  const mockDeriveErrorMessage = vi.fn((errorData: any) => {
    if (typeof errorData === "string") return errorData;
    return errorData?.message || errorData?.error || "An error occurred";
  });
  const mockHandleError = vi.fn();

  return {
    mockProxyBaseUrl,
    mockAccessToken,
    mockHeaderName,
    mockProxyConfigResponse,
    mockDeleteResponse,
    mockUseAuthorized,
    mockGetGlobalLitellmHeaderName,
    mockDeriveErrorMessage,
    mockHandleError,
  };
});

vi.mock("../useAuthorized", () => ({
  default: () => mockUseAuthorized(),
}));

vi.mock("@/components/networking", () => ({
  proxyBaseUrl: mockProxyBaseUrl,
  getGlobalLitellmHeaderName: mockGetGlobalLitellmHeaderName,
  deriveErrorMessage: mockDeriveErrorMessage,
  handleError: mockHandleError,
}));

vi.mock("../common/queryKeysFactory", () => ({
  createQueryKeys: vi.fn((resource: string) => ({
    all: [resource],
    lists: () => [resource, "list"],
    list: (params?: any) => [resource, "list", { params }],
    details: () => [resource, "detail"],
    detail: (uid: string) => [resource, "detail", uid],
  })),
}));

describe("useProxyConfig", () => {
  let queryClient: QueryClient;
  let fetchSpy: ReturnType<typeof vi.fn>;

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
      accessToken: mockAccessToken,
      userId: "test-user-id",
      userRole: "Admin",
      token: "test-token",
      userEmail: "test@example.com",
      premiumUser: false,
      disabledPersonalKeyCreation: null,
      showSSOBanner: false,
    });

    fetchSpy = vi.fn();
    global.fetch = fetchSpy;
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  const wrapper = ({ children }: { children: ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);

  it("should render successfully", () => {
    (fetchSpy as any).mockResolvedValue({
      ok: true,
      json: async () => mockProxyConfigResponse,
    });

    const { result } = renderHook(() => useProxyConfig(ConfigType.GENERAL_SETTINGS), { wrapper });

    expect(result.current).toBeDefined();
    expect(result.current.isLoading).toBe(true);
  });

  it("should return proxy config data when query is successful", async () => {
    (fetchSpy as any).mockResolvedValue({
      ok: true,
      json: async () => mockProxyConfigResponse,
    });

    const { result } = renderHook(() => useProxyConfig(ConfigType.GENERAL_SETTINGS), { wrapper });

    expect(result.current.isLoading).toBe(true);
    expect(result.current.data).toBeUndefined();

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual(mockProxyConfigResponse);
    expect(result.current.error).toBeNull();
    expect(fetchSpy).toHaveBeenCalledWith(
      `${mockProxyBaseUrl}/config/list?config_type=${ConfigType.GENERAL_SETTINGS}`,
      {
        method: "GET",
        headers: {
          [mockHeaderName]: `Bearer ${mockAccessToken}`,
          "Content-Type": "application/json",
        },
      },
    );
    expect(fetchSpy).toHaveBeenCalledTimes(1);
  });

  it("should handle error when API call fails", async () => {
    const errorMessage = "Failed to fetch proxy config";
    const errorResponse = { message: errorMessage };

    (fetchSpy as any).mockResolvedValue({
      ok: false,
      json: async () => errorResponse,
    });

    const { result } = renderHook(() => useProxyConfig(ConfigType.GENERAL_SETTINGS), { wrapper });

    expect(result.current.isLoading).toBe(true);

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toBeDefined();
    expect(result.current.data).toBeUndefined();
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

    const { result } = renderHook(() => useProxyConfig(ConfigType.GENERAL_SETTINGS), { wrapper });

    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);

    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("should use correct query key with config type filter", async () => {
    (fetchSpy as any).mockResolvedValue({
      ok: true,
      json: async () => mockProxyConfigResponse,
    });

    const { result } = renderHook(() => useProxyConfig(ConfigType.GENERAL_SETTINGS), { wrapper });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual(mockProxyConfigResponse);
  });

  it("should handle network errors", async () => {
    const networkError = new Error("Network error");
    (fetchSpy as any).mockRejectedValue(networkError);

    const { result } = renderHook(() => useProxyConfig(ConfigType.GENERAL_SETTINGS), { wrapper });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toBeDefined();
    expect(result.current.data).toBeUndefined();
  });

  it("should handle empty config response", async () => {
    (fetchSpy as any).mockResolvedValue({
      ok: true,
      json: async () => [],
    });

    const { result } = renderHook(() => useProxyConfig(ConfigType.GENERAL_SETTINGS), { wrapper });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual([]);
  });
});

describe("useDeleteProxyConfigField", () => {
  let queryClient: QueryClient;
  let fetchSpy: ReturnType<typeof vi.fn>;

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
      accessToken: mockAccessToken,
      userId: "test-user-id",
      userRole: "Admin",
      token: "test-token",
      userEmail: "test@example.com",
      premiumUser: false,
      disabledPersonalKeyCreation: null,
      showSSOBanner: false,
    });

    fetchSpy = vi.fn();
    global.fetch = fetchSpy;
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  const wrapper = ({ children }: { children: ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);

  it("should render successfully", () => {
    const { result } = renderHook(() => useDeleteProxyConfigField(), { wrapper });

    expect(result.current).toBeDefined();
    expect(result.current.isIdle).toBe(true);
  });

  it("should successfully delete a proxy config field", async () => {
    (fetchSpy as any).mockResolvedValue({
      ok: true,
      json: async () => mockDeleteResponse,
    });

    const { result } = renderHook(() => useDeleteProxyConfigField(), { wrapper });

    const deleteRequest: DeleteProxyConfigFieldRequest = {
      config_type: ConfigType.GENERAL_SETTINGS,
      field_name: GeneralSettingsFieldName.MAXIMUM_SPEND_LOGS_RETENTION_PERIOD,
    };

    result.current.mutate(deleteRequest);

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual(mockDeleteResponse);
    expect(fetchSpy).toHaveBeenCalledWith(`${mockProxyBaseUrl}/config/field/delete`, {
      method: "POST",
      headers: {
        [mockHeaderName]: `Bearer ${mockAccessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(deleteRequest),
    });
    expect(fetchSpy).toHaveBeenCalledTimes(1);
  });

  it("should handle error when delete request fails", async () => {
    const errorMessage = "Failed to delete field";
    const errorResponse = { message: errorMessage };

    (fetchSpy as any).mockResolvedValue({
      ok: false,
      json: async () => errorResponse,
    });

    const { result } = renderHook(() => useDeleteProxyConfigField(), { wrapper });

    const deleteRequest: DeleteProxyConfigFieldRequest = {
      config_type: ConfigType.GENERAL_SETTINGS,
      field_name: GeneralSettingsFieldName.MAXIMUM_SPEND_LOGS_RETENTION_PERIOD,
    };

    result.current.mutate(deleteRequest);

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toBeDefined();
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

    const { result } = renderHook(() => useDeleteProxyConfigField(), { wrapper });

    const deleteRequest: DeleteProxyConfigFieldRequest = {
      config_type: ConfigType.GENERAL_SETTINGS,
      field_name: GeneralSettingsFieldName.MAXIMUM_SPEND_LOGS_RETENTION_PERIOD,
    };

    result.current.mutate(deleteRequest);

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error?.message).toBe("Access token is required");
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("should handle network errors during delete", async () => {
    const networkError = new Error("Network error");
    (fetchSpy as any).mockRejectedValue(networkError);

    const { result } = renderHook(() => useDeleteProxyConfigField(), { wrapper });

    const deleteRequest: DeleteProxyConfigFieldRequest = {
      config_type: ConfigType.GENERAL_SETTINGS,
      field_name: GeneralSettingsFieldName.MAXIMUM_SPEND_LOGS_RETENTION_PERIOD,
    };

    result.current.mutate(deleteRequest);

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toBeDefined();
  });
});

describe("getProxyConfigCall", () => {
  let fetchSpy: ReturnType<typeof vi.fn>;
  let consoleErrorSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    fetchSpy = vi.fn();
    global.fetch = fetchSpy;
    consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("should successfully fetch proxy config", async () => {
    (fetchSpy as any).mockResolvedValue({
      ok: true,
      json: async () => mockProxyConfigResponse,
    });

    const result = await getProxyConfigCall(mockAccessToken, ConfigType.GENERAL_SETTINGS);

    expect(result).toEqual(mockProxyConfigResponse);
    expect(fetchSpy).toHaveBeenCalledWith(
      `${mockProxyBaseUrl}/config/list?config_type=${ConfigType.GENERAL_SETTINGS}`,
      {
        method: "GET",
        headers: {
          [mockHeaderName]: `Bearer ${mockAccessToken}`,
          "Content-Type": "application/json",
        },
      },
    );
  });

  it("should throw error when API returns error response", async () => {
    const errorMessage = "Failed to fetch config";
    const errorResponse = { message: errorMessage };

    (fetchSpy as any).mockResolvedValue({
      ok: false,
      json: async () => errorResponse,
    });

    await expect(getProxyConfigCall(mockAccessToken, ConfigType.GENERAL_SETTINGS)).rejects.toThrow(errorMessage);
  });

  it("should handle network errors", async () => {
    const networkError = new Error("Network error");
    (fetchSpy as any).mockRejectedValue(networkError);

    await expect(getProxyConfigCall(mockAccessToken, ConfigType.GENERAL_SETTINGS)).rejects.toThrow("Network error");
    expect(consoleErrorSpy).toHaveBeenCalled();
  });
});

describe("deleteProxyConfigFieldCall", () => {
  let fetchSpy: ReturnType<typeof vi.fn>;
  let consoleErrorSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    fetchSpy = vi.fn();
    global.fetch = fetchSpy;
    consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("should successfully delete proxy config field", async () => {
    (fetchSpy as any).mockResolvedValue({
      ok: true,
      json: async () => mockDeleteResponse,
    });

    const deleteRequest: DeleteProxyConfigFieldRequest = {
      config_type: ConfigType.GENERAL_SETTINGS,
      field_name: GeneralSettingsFieldName.MAXIMUM_SPEND_LOGS_RETENTION_PERIOD,
    };

    const result = await deleteProxyConfigFieldCall(mockAccessToken, deleteRequest);

    expect(result).toEqual(mockDeleteResponse);
    expect(fetchSpy).toHaveBeenCalledWith(`${mockProxyBaseUrl}/config/field/delete`, {
      method: "POST",
      headers: {
        [mockHeaderName]: `Bearer ${mockAccessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(deleteRequest),
    });
  });

  it("should throw error when API returns error response", async () => {
    const errorMessage = "Failed to delete field";
    const errorResponse = { message: errorMessage };

    (fetchSpy as any).mockResolvedValue({
      ok: false,
      json: async () => errorResponse,
    });

    const deleteRequest: DeleteProxyConfigFieldRequest = {
      config_type: ConfigType.GENERAL_SETTINGS,
      field_name: GeneralSettingsFieldName.MAXIMUM_SPEND_LOGS_RETENTION_PERIOD,
    };

    await expect(deleteProxyConfigFieldCall(mockAccessToken, deleteRequest)).rejects.toThrow(errorMessage);
  });

  it("should handle network errors", async () => {
    const networkError = new Error("Network error");
    (fetchSpy as any).mockRejectedValue(networkError);

    const deleteRequest: DeleteProxyConfigFieldRequest = {
      config_type: ConfigType.GENERAL_SETTINGS,
      field_name: GeneralSettingsFieldName.MAXIMUM_SPEND_LOGS_RETENTION_PERIOD,
    };

    await expect(deleteProxyConfigFieldCall(mockAccessToken, deleteRequest)).rejects.toThrow("Network error");
    expect(consoleErrorSpy).toHaveBeenCalled();
  });
});
