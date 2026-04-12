import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React, { ReactNode } from "react";
import { useProjects, ProjectResponse } from "./useProjects";

vi.mock("@/components/networking", () => ({
  getProxyBaseUrl: vi.fn(() => ""),
  getGlobalLitellmHeaderName: vi.fn(() => "Authorization"),
  deriveErrorMessage: vi.fn((data: any) => data?.error || "Error"),
  handleError: vi.fn(),
}));

const mockUseAuthorized = vi.fn();
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => mockUseAuthorized(),
}));

const mockProjects: ProjectResponse[] = [
  {
    project_id: "proj-1",
    project_alias: "Test Project",
    description: "A test project",
    team_id: "team-1",
    budget_id: null,
    metadata: null,
    models: ["gpt-4"],
    spend: 25.0,
    model_spend: null,
    model_rpm_limit: null,
    model_tpm_limit: null,
    blocked: false,
    object_permission_id: null,
    created_at: "2024-01-01T00:00:00Z",
    created_by: "user-1",
    updated_at: "2024-01-02T00:00:00Z",
    updated_by: "user-1",
    litellm_budget_table: null,
  },
  {
    project_id: "proj-2",
    project_alias: "Test Project 2",
    description: null,
    team_id: "team-1",
    budget_id: null,
    metadata: null,
    models: [],
    spend: 0,
    model_spend: null,
    model_rpm_limit: null,
    model_tpm_limit: null,
    blocked: false,
    object_permission_id: null,
    created_at: "2024-01-03T00:00:00Z",
    created_by: "user-1",
    updated_at: "2024-01-03T00:00:00Z",
    updated_by: "user-1",
    litellm_budget_table: null,
  },
];

function makeWrapper(queryClient: QueryClient) {
  return ({ children }: { children: ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);
}

describe("useProjects", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    vi.clearAllMocks();
    global.fetch = vi.fn();
    mockUseAuthorized.mockReturnValue({ accessToken: "test-token", userRole: "Admin" });
  });

  it("should render", () => {
    (global.fetch as any).mockResolvedValue({ ok: true, json: async () => mockProjects });
    const { result } = renderHook(() => useProjects(), { wrapper: makeWrapper(queryClient) });
    expect(result.current).toBeDefined();
  });

  it("should return projects when the request succeeds", async () => {
    (global.fetch as any).mockResolvedValue({ ok: true, json: async () => mockProjects });
    const { result } = renderHook(() => useProjects(), { wrapper: makeWrapper(queryClient) });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toEqual(mockProjects);
  });

  it("should call GET /project/list with the auth header", async () => {
    (global.fetch as any).mockResolvedValue({ ok: true, json: async () => mockProjects });
    renderHook(() => useProjects(), { wrapper: makeWrapper(queryClient) });
    await waitFor(() => expect(global.fetch).toHaveBeenCalled());
    const [url, init] = (global.fetch as any).mock.calls[0];
    expect(url).toContain("/project/list");
    expect(init.headers["Authorization"]).toBe("Bearer test-token");
  });

  it("should set isError when the request fails", async () => {
    (global.fetch as any).mockResolvedValue({
      ok: false,
      json: async () => ({ error: "Not authorized" }),
    });
    const { result } = renderHook(() => useProjects(), { wrapper: makeWrapper(queryClient) });
    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(result.current.data).toBeUndefined();
  });

  it("should not fetch when accessToken is missing", () => {
    mockUseAuthorized.mockReturnValue({ accessToken: null, userRole: "Admin" });
    const { result } = renderHook(() => useProjects(), { wrapper: makeWrapper(queryClient) });
    expect(result.current.isFetched).toBe(false);
    expect(global.fetch).not.toHaveBeenCalled();
  });

  it("should not fetch when userRole is not an admin role", () => {
    mockUseAuthorized.mockReturnValue({ accessToken: "test-token", userRole: "Internal User" });
    const { result } = renderHook(() => useProjects(), { wrapper: makeWrapper(queryClient) });
    expect(result.current.isFetched).toBe(false);
    expect(global.fetch).not.toHaveBeenCalled();
  });
});
