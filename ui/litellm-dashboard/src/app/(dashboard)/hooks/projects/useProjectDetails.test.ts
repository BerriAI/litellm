import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React, { ReactNode } from "react";
import { useProjectDetails } from "./useProjectDetails";
import { projectKeys, ProjectResponse } from "./useProjects";

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

const mockProject: ProjectResponse = {
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
};

const mockProjects: ProjectResponse[] = [
  mockProject,
  { ...mockProject, project_id: "proj-2", project_alias: "Test Project 2" },
];

function makeWrapper(queryClient: QueryClient) {
  return ({ children }: { children: ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);
}

describe("useProjectDetails", () => {
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
    (global.fetch as any).mockResolvedValue({ ok: true, json: async () => mockProject });
    const { result } = renderHook(() => useProjectDetails("proj-1"), {
      wrapper: makeWrapper(queryClient),
    });
    expect(result.current).toBeDefined();
  });

  it("should return project details when the request succeeds", async () => {
    (global.fetch as any).mockResolvedValue({ ok: true, json: async () => mockProject });
    const { result } = renderHook(() => useProjectDetails("proj-1"), {
      wrapper: makeWrapper(queryClient),
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toEqual(mockProject);
  });

  it("should call /project/info with the projectId encoded as a query param", async () => {
    (global.fetch as any).mockResolvedValue({ ok: true, json: async () => mockProject });
    renderHook(() => useProjectDetails("proj-1"), { wrapper: makeWrapper(queryClient) });
    await waitFor(() => expect(global.fetch).toHaveBeenCalled());
    const [url] = (global.fetch as any).mock.calls[0];
    expect(url).toContain("/project/info");
    expect(url).toContain("project_id=proj-1");
  });

  it("should not fetch when projectId is missing", () => {
    const { result } = renderHook(() => useProjectDetails(undefined), {
      wrapper: makeWrapper(queryClient),
    });
    expect(result.current.isFetched).toBe(false);
    expect(global.fetch).not.toHaveBeenCalled();
  });

  it("should not fetch when accessToken is missing", () => {
    mockUseAuthorized.mockReturnValue({ accessToken: null, userRole: "Admin" });
    const { result } = renderHook(() => useProjectDetails("proj-1"), {
      wrapper: makeWrapper(queryClient),
    });
    expect(result.current.isFetched).toBe(false);
    expect(global.fetch).not.toHaveBeenCalled();
  });

  it("should not fetch when userRole is not an admin role", () => {
    mockUseAuthorized.mockReturnValue({ accessToken: "test-token", userRole: "Internal User" });
    const { result } = renderHook(() => useProjectDetails("proj-1"), {
      wrapper: makeWrapper(queryClient),
    });
    expect(result.current.isFetched).toBe(false);
    expect(global.fetch).not.toHaveBeenCalled();
  });

  it("should seed initialData from the projects list cache", async () => {
    (global.fetch as any).mockResolvedValue({ ok: true, json: async () => mockProject });
    queryClient.setQueryData(projectKeys.list({}), mockProjects);
    const { result } = renderHook(() => useProjectDetails("proj-1"), {
      wrapper: makeWrapper(queryClient),
    });
    expect(result.current.data).toEqual(mockProject);
    expect(result.current.isLoading).toBe(false);
    await waitFor(() => expect(result.current.isFetching).toBe(false));
  });

  it("should return undefined initialData when projectId is not in the cache", () => {
    queryClient.setQueryData(projectKeys.list({}), mockProjects);
    const { result } = renderHook(() => useProjectDetails("non-existent"), {
      wrapper: makeWrapper(queryClient),
    });
    expect(result.current.data).toBeUndefined();
  });

  it("should set isError when the request fails", async () => {
    (global.fetch as any).mockResolvedValue({
      ok: false,
      json: async () => ({ error: "Not found" }),
    });
    const { result } = renderHook(() => useProjectDetails("proj-1"), {
      wrapper: makeWrapper(queryClient),
    });
    await waitFor(() => expect(result.current.isError).toBe(true));
  });
});
