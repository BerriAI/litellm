import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import React, { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  useModelsInfo,
  useModelHub,
  useAllProxyModels,
  useSelectedTeamModels,
  type ProxyModel,
  type AllProxyModelsResponse,
  type PaginatedModelInfoResponse,
} from "./useModels";

vi.mock("@/components/networking", () => ({
  modelInfoCall: vi.fn(),
  modelHubCall: vi.fn(),
  modelAvailableCall: vi.fn(),
}));

const mockUseAuthorized = vi.fn();
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => mockUseAuthorized(),
}));

import { modelInfoCall, modelHubCall, modelAvailableCall } from "@/components/networking";

const mockProxyModel: ProxyModel = {
  id: "model-1",
  object: "model",
  created: 1234567890,
  owned_by: "openai",
};

const mockPaginatedModelInfoResponse: PaginatedModelInfoResponse = {
  data: [{ id: "model-1", name: "Test Model" }],
  total_count: 1,
  current_page: 1,
  total_pages: 1,
  size: 50,
};

const mockAllProxyModelsResponse: AllProxyModelsResponse = {
  data: [mockProxyModel],
};

describe("useModelsInfo", () => {
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

  it("should render without crashing", () => {
    (modelInfoCall as any).mockResolvedValue(mockPaginatedModelInfoResponse);

    const { result } = renderHook(() => useModelsInfo(), { wrapper });

    expect(result.current).toBeDefined();
  });

  it("should return models data when query is successful", async () => {
    (modelInfoCall as any).mockResolvedValue(mockPaginatedModelInfoResponse);

    const { result } = renderHook(() => useModelsInfo(), { wrapper });

    expect(result.current.isLoading).toBe(true);
    expect(result.current.data).toBeUndefined();

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual(mockPaginatedModelInfoResponse);
    expect(result.current.error).toBeNull();
    expect(modelInfoCall).toHaveBeenCalledWith(
      "test-access-token",
      "test-user-id",
      "Admin",
      1,
      50,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined
    );
    expect(modelInfoCall).toHaveBeenCalledTimes(1);
  });

  it("should use custom page and size parameters", async () => {
    (modelInfoCall as any).mockResolvedValue(mockPaginatedModelInfoResponse);

    const { result } = renderHook(() => useModelsInfo(2, 25), { wrapper });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(modelInfoCall).toHaveBeenCalledWith(
      "test-access-token",
      "test-user-id",
      "Admin",
      2,
      25,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined
    );
  });

  it("should handle error when modelInfoCall fails", async () => {
    const errorMessage = "Failed to fetch models";
    const testError = new Error(errorMessage);

    (modelInfoCall as any).mockRejectedValue(testError);

    const { result } = renderHook(() => useModelsInfo(), { wrapper });

    expect(result.current.isLoading).toBe(true);

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(testError);
    expect(result.current.data).toBeUndefined();
    expect(modelInfoCall).toHaveBeenCalledTimes(1);
  });

  it("should not execute query when accessToken is missing", () => {
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

    const { result } = renderHook(() => useModelsInfo(), { wrapper });

    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);
    expect(modelInfoCall).not.toHaveBeenCalled();
  });

  it("should not execute query when userId is missing", () => {
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

    const { result } = renderHook(() => useModelsInfo(), { wrapper });

    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);
    expect(modelInfoCall).not.toHaveBeenCalled();
  });

  it("should not execute query when userRole is missing", () => {
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

    const { result } = renderHook(() => useModelsInfo(), { wrapper });

    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);
    expect(modelInfoCall).not.toHaveBeenCalled();
  });

  it("should not execute query when all required auth values are missing", () => {
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

    const { result } = renderHook(() => useModelsInfo(), { wrapper });

    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);
    expect(modelInfoCall).not.toHaveBeenCalled();
  });
});

describe("useModelHub", () => {
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

  it("should render without crashing", () => {
    (modelHubCall as any).mockResolvedValue({ data: [] });

    const { result } = renderHook(() => useModelHub(), { wrapper });

    expect(result.current).toBeDefined();
  });

  it("should return model hub data when query is successful", async () => {
    const mockHubData = { data: [{ id: "hub-1", name: "Test Hub" }] };
    (modelHubCall as any).mockResolvedValue(mockHubData);

    const { result } = renderHook(() => useModelHub(), { wrapper });

    expect(result.current.isLoading).toBe(true);
    expect(result.current.data).toBeUndefined();

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual(mockHubData);
    expect(result.current.error).toBeNull();
    expect(modelHubCall).toHaveBeenCalledWith("test-access-token");
    expect(modelHubCall).toHaveBeenCalledTimes(1);
  });

  it("should handle error when modelHubCall fails", async () => {
    const errorMessage = "Failed to fetch model hub";
    const testError = new Error(errorMessage);

    (modelHubCall as any).mockRejectedValue(testError);

    const { result } = renderHook(() => useModelHub(), { wrapper });

    expect(result.current.isLoading).toBe(true);

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(testError);
    expect(result.current.data).toBeUndefined();
    expect(modelHubCall).toHaveBeenCalledTimes(1);
  });

  it("should not execute query when accessToken is missing", () => {
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

    const { result } = renderHook(() => useModelHub(), { wrapper });

    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);
    expect(modelHubCall).not.toHaveBeenCalled();
  });
});

describe("useAllProxyModels", () => {
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

  it("should render without crashing", () => {
    (modelAvailableCall as any).mockResolvedValue(mockAllProxyModelsResponse);

    const { result } = renderHook(() => useAllProxyModels(), { wrapper });

    expect(result.current).toBeDefined();
  });

  it("should return all proxy models data when query is successful", async () => {
    (modelAvailableCall as any).mockResolvedValue(mockAllProxyModelsResponse);

    const { result } = renderHook(() => useAllProxyModels(), { wrapper });

    expect(result.current.isLoading).toBe(true);
    expect(result.current.data).toBeUndefined();

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual(mockAllProxyModelsResponse);
    expect(result.current.error).toBeNull();
    expect(modelAvailableCall).toHaveBeenCalledWith(
      "test-access-token",
      "test-user-id",
      "Admin",
      true,
      null,
      true,
      false,
      "expand"
    );
    expect(modelAvailableCall).toHaveBeenCalledTimes(1);
  });

  it("should handle error when modelAvailableCall fails", async () => {
    const errorMessage = "Failed to fetch proxy models";
    const testError = new Error(errorMessage);

    (modelAvailableCall as any).mockRejectedValue(testError);

    const { result } = renderHook(() => useAllProxyModels(), { wrapper });

    expect(result.current.isLoading).toBe(true);

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(testError);
    expect(result.current.data).toBeUndefined();
    expect(modelAvailableCall).toHaveBeenCalledTimes(1);
  });

  it("should not execute query when accessToken is missing", () => {
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

    const { result } = renderHook(() => useAllProxyModels(), { wrapper });

    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);
    expect(modelAvailableCall).not.toHaveBeenCalled();
  });

  it("should not execute query when userId is missing", () => {
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

    const { result } = renderHook(() => useAllProxyModels(), { wrapper });

    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);
    expect(modelAvailableCall).not.toHaveBeenCalled();
  });

  it("should not execute query when userRole is missing", () => {
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

    const { result } = renderHook(() => useAllProxyModels(), { wrapper });

    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);
    expect(modelAvailableCall).not.toHaveBeenCalled();
  });
});

describe("useSelectedTeamModels", () => {
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

  it("should render without crashing", () => {
    (modelAvailableCall as any).mockResolvedValue(mockAllProxyModelsResponse);

    const { result } = renderHook(() => useSelectedTeamModels("team-1"), { wrapper });

    expect(result.current).toBeDefined();
  });

  it("should return team models data when query is successful", async () => {
    (modelAvailableCall as any).mockResolvedValue(mockAllProxyModelsResponse);

    const { result } = renderHook(() => useSelectedTeamModels("team-1"), { wrapper });

    expect(result.current.isLoading).toBe(true);
    expect(result.current.data).toBeUndefined();

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual(mockAllProxyModelsResponse);
    expect(result.current.error).toBeNull();
    expect(modelAvailableCall).toHaveBeenCalledWith(
      "test-access-token",
      "test-user-id",
      "Admin",
      true,
      "team-1"
    );
    expect(modelAvailableCall).toHaveBeenCalledTimes(1);
  });

  it("should handle error when modelAvailableCall fails", async () => {
    const errorMessage = "Failed to fetch team models";
    const testError = new Error(errorMessage);

    (modelAvailableCall as any).mockRejectedValue(testError);

    const { result } = renderHook(() => useSelectedTeamModels("team-1"), { wrapper });

    expect(result.current.isLoading).toBe(true);

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toEqual(testError);
    expect(result.current.data).toBeUndefined();
    expect(modelAvailableCall).toHaveBeenCalledTimes(1);
  });

  it("should not execute query when teamID is null", () => {
    const { result } = renderHook(() => useSelectedTeamModels(null), { wrapper });

    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);
    expect(modelAvailableCall).not.toHaveBeenCalled();
  });

  it("should not execute query when accessToken is missing", () => {
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

    const { result } = renderHook(() => useSelectedTeamModels("team-1"), { wrapper });

    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);
    expect(modelAvailableCall).not.toHaveBeenCalled();
  });

  it("should not execute query when userId is missing", () => {
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

    const { result } = renderHook(() => useSelectedTeamModels("team-1"), { wrapper });

    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);
    expect(modelAvailableCall).not.toHaveBeenCalled();
  });

  it("should not execute query when userRole is missing", () => {
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

    const { result } = renderHook(() => useSelectedTeamModels("team-1"), { wrapper });

    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);
    expect(modelAvailableCall).not.toHaveBeenCalled();
  });

  it("should not execute query when teamID is missing and other auth values are present", () => {
    const { result } = renderHook(() => useSelectedTeamModels(null), { wrapper });

    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toBeUndefined();
    expect(result.current.isFetched).toBe(false);
    expect(modelAvailableCall).not.toHaveBeenCalled();
  });
});
