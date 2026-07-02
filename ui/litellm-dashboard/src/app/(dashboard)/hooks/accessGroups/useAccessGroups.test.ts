import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React, { ReactNode } from "react";
import { useAccessGroups } from "./useAccessGroups";
import type { paths } from "@/lib/http/schema";

type AccessGroupListResponse = paths["/v1/access_group"]["get"]["responses"][200]["content"]["application/json"];

const { fetchMock } = vi.hoisted(() => {
  const fetchMock = vi.fn();
  globalThis.fetch = fetchMock as unknown as typeof fetch;
  return { fetchMock };
});

vi.mock("@/components/networking", () => ({
  getProxyBaseUrl: () => "http://localhost:4000",
  getGlobalLitellmHeaderName: () => "Authorization",
}));

const mockUseAuthorized = vi.fn();
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => mockUseAuthorized(),
}));

const DEFAULT_AUTH = { accessToken: "test-access-token", userRole: "Admin" };

const requestOf = (arg: unknown): Request => arg as Request;
const jsonResponse = (body: unknown, status = 200): Response =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

const groups: AccessGroupListResponse = [
  {
    access_group_id: "ag-1",
    access_group_name: "Group One",
    description: null,
    access_model_names: [],
    access_mcp_server_ids: [],
    access_agent_ids: [],
    assigned_team_ids: [],
    assigned_key_ids: [],
    created_at: "2025-01-01T00:00:00Z",
    created_by: null,
    updated_at: "2025-01-01T00:00:00Z",
    updated_by: null,
  },
];

describe("useAccessGroups", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    fetchMock.mockReset();
    mockUseAuthorized.mockReset();
    mockUseAuthorized.mockReturnValue(DEFAULT_AUTH);
  });

  const wrapper = ({ children }: { children: ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);

  it("fetches GET /v1/access_group with the bearer header and returns typed data", async () => {
    fetchMock.mockResolvedValue(jsonResponse(groups));
    const { result } = renderHook(() => useAccessGroups(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    const request = requestOf(fetchMock.mock.calls[0][0]);
    expect(new URL(request.url).pathname).toBe("/v1/access_group");
    expect(request.method).toBe("GET");
    expect(request.headers.get("Authorization")).toBe("Bearer test-access-token");
    expect(result.current.data?.[0].access_group_id).toBe("ag-1");
  });

  it.each([
    ["null token", { accessToken: null }],
    ["non-admin role", { userRole: "Internal User" }],
  ])("does not fire a request when gated by %s", (_label, override) => {
    fetchMock.mockResolvedValue(jsonResponse(groups));
    mockUseAuthorized.mockReturnValue({ ...DEFAULT_AUTH, ...override });

    const { result } = renderHook(() => useAccessGroups(), { wrapper });

    expect(result.current.isLoading).toBe(false);
    expect(result.current.isFetched).toBe(false);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
