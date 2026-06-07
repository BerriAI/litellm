import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React, { ReactNode } from "react";
import { useCreateAccessGroup } from "./useCreateAccessGroup";

const { fetchMock } = vi.hoisted(() => {
  const fetchMock = vi.fn();
  globalThis.fetch = fetchMock as unknown as typeof fetch;
  return { fetchMock };
});

const handleError = vi.fn();
const deriveErrorMessage = vi.fn(() => "derived message");
vi.mock("@/components/networking", () => ({
  getProxyBaseUrl: () => "http://localhost:4000",
  getGlobalLitellmHeaderName: () => "Authorization",
  handleError: (...args: unknown[]) => handleError(...args),
  deriveErrorMessage: (...args: unknown[]) => deriveErrorMessage(...args),
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({ accessToken: "test-access-token" }),
}));

const requestOf = (arg: unknown): Request => arg as Request;
const jsonResponse = (body: unknown, status = 200): Response =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

const created = { access_group_id: "ag-new", access_group_name: "New Group" };

describe("useCreateAccessGroup", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({ defaultOptions: { mutations: { retry: false } } });
    fetchMock.mockReset();
    handleError.mockReset();
    deriveErrorMessage.mockClear();
  });

  const wrapper = ({ children }: { children: ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);

  it("POSTs the body to /v1/access_group with the bearer header", async () => {
    fetchMock.mockResolvedValue(jsonResponse(created, 201));
    const { result } = renderHook(() => useCreateAccessGroup(), { wrapper });

    result.current.mutate({ access_group_name: "New Group", access_model_names: ["gpt-4o"] });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    const request = requestOf(fetchMock.mock.calls[0][0]);
    expect(new URL(request.url).pathname).toBe("/v1/access_group");
    expect(request.method).toBe("POST");
    expect(request.headers.get("Authorization")).toBe("Bearer test-access-token");
    expect(await request.json()).toEqual({ access_group_name: "New Group", access_model_names: ["gpt-4o"] });
  });

  it("invalidates the access-group list on success", async () => {
    const invalidate = vi.spyOn(queryClient, "invalidateQueries");
    fetchMock.mockResolvedValue(jsonResponse(created, 201));
    const { result } = renderHook(() => useCreateAccessGroup(), { wrapper });

    result.current.mutate({ access_group_name: "New Group" });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(invalidate).toHaveBeenCalledWith({ queryKey: ["get", "/v1/access_group"] });
  });

  it("routes a failed create through the global error handler", async () => {
    fetchMock.mockResolvedValue(jsonResponse({ detail: "boom" }, 400));
    const { result } = renderHook(() => useCreateAccessGroup(), { wrapper });

    result.current.mutate({ access_group_name: "New Group" });

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(deriveErrorMessage).toHaveBeenCalled();
    expect(handleError).toHaveBeenCalledWith("derived message");
  });
});
