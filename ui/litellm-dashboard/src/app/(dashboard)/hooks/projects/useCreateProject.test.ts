import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React, { ReactNode } from "react";
import { useCreateProject, ProjectCreateParams } from "./useCreateProject";
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

function makeWrapper(queryClient: QueryClient) {
  return ({ children }: { children: ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);
}

describe("useCreateProject", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
    });
    vi.clearAllMocks();
    global.fetch = vi.fn();
    mockUseAuthorized.mockReturnValue({ accessToken: "test-token", userRole: "Admin" });
  });

  it("should render", () => {
    const { result } = renderHook(() => useCreateProject(), {
      wrapper: makeWrapper(queryClient),
    });
    expect(result.current.mutate).toBeDefined();
  });

  it("should POST to /project/new and return the created project", async () => {
    (global.fetch as any).mockResolvedValue({ ok: true, json: async () => mockProject });
    const { result } = renderHook(() => useCreateProject(), {
      wrapper: makeWrapper(queryClient),
    });
    const params: ProjectCreateParams = { team_id: "team-1", project_alias: "New Project" };
    const data = await result.current.mutateAsync(params);
    expect(data).toEqual(mockProject);
    const [url, init] = (global.fetch as any).mock.calls[0];
    expect(url).toContain("/project/new");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toMatchObject(params);
  });

  it("should invalidate project queries on success", async () => {
    (global.fetch as any).mockResolvedValue({ ok: true, json: async () => mockProject });
    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries");
    const { result } = renderHook(() => useCreateProject(), {
      wrapper: makeWrapper(queryClient),
    });
    await result.current.mutateAsync({ team_id: "team-1" });
    expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: projectKeys.all });
  });

  it("should set isError when the request fails", async () => {
    (global.fetch as any).mockResolvedValue({
      ok: false,
      json: async () => ({ error: "Server error" }),
    });
    const { result } = renderHook(() => useCreateProject(), {
      wrapper: makeWrapper(queryClient),
    });
    result.current.mutateAsync({ team_id: "team-1" }).catch(() => {});
    await waitFor(() => expect(result.current.isError).toBe(true));
  });

  it("should throw when accessToken is missing", async () => {
    mockUseAuthorized.mockReturnValue({ accessToken: null, userRole: "Admin" });
    const { result } = renderHook(() => useCreateProject(), {
      wrapper: makeWrapper(queryClient),
    });
    await expect(result.current.mutateAsync({ team_id: "team-1" })).rejects.toThrow(
      "Access token is required"
    );
    expect(global.fetch).not.toHaveBeenCalled();
  });
});
