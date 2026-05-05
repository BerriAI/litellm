import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React, { ReactNode } from "react";
import { useDeleteProject } from "./useDeleteProject";
import { projectKeys } from "./useProjects";

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

function makeWrapper(queryClient: QueryClient) {
  return ({ children }: { children: ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);
}

describe("useDeleteProject", () => {
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
    const { result } = renderHook(() => useDeleteProject(), {
      wrapper: makeWrapper(queryClient),
    });
    expect(result.current.mutate).toBeDefined();
  });

  it("should send DELETE to /project/delete with the given project IDs", async () => {
    (global.fetch as any).mockResolvedValue({ ok: true, json: async () => ({}) });
    const { result } = renderHook(() => useDeleteProject(), {
      wrapper: makeWrapper(queryClient),
    });
    await result.current.mutateAsync(["proj-1", "proj-2"]);
    const [url, init] = (global.fetch as any).mock.calls[0];
    expect(url).toContain("/project/delete");
    expect(init.method).toBe("DELETE");
    expect(JSON.parse(init.body)).toEqual({ project_ids: ["proj-1", "proj-2"] });
  });

  it("should invalidate project queries on success", async () => {
    (global.fetch as any).mockResolvedValue({ ok: true, json: async () => ({}) });
    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries");
    const { result } = renderHook(() => useDeleteProject(), {
      wrapper: makeWrapper(queryClient),
    });
    await result.current.mutateAsync(["proj-1"]);
    expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: projectKeys.all });
  });

  it("should set isError when the request fails", async () => {
    (global.fetch as any).mockResolvedValue({
      ok: false,
      json: async () => ({ error: "Not found" }),
    });
    const { result } = renderHook(() => useDeleteProject(), {
      wrapper: makeWrapper(queryClient),
    });
    result.current.mutateAsync(["proj-1"]).catch(() => {});
    await waitFor(() => expect(result.current.isError).toBe(true));
  });

  it("should throw when accessToken is missing", async () => {
    mockUseAuthorized.mockReturnValue({ accessToken: null, userRole: "Admin" });
    const { result } = renderHook(() => useDeleteProject(), {
      wrapper: makeWrapper(queryClient),
    });
    await expect(result.current.mutateAsync(["proj-1"])).rejects.toThrow(
      "Access token is required"
    );
    expect(global.fetch).not.toHaveBeenCalled();
  });
});
