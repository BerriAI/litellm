import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React, { ReactNode } from "react";
import { useSetKeyBlockedState, setKeyBlockedState } from "./useSetKeyBlockedState";
import { apiClient } from "@/components/networking";

vi.mock("@/components/networking", () => ({
  apiClient: { post: vi.fn() },
}));

const mockUseAuthorized = vi.fn();
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => mockUseAuthorized(),
}));

const mockPost = vi.mocked(apiClient.post);

const createWrapper = () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  const wrapper = ({ children }: { children: ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);
  return { queryClient, wrapper };
};

describe("setKeyBlockedState", () => {
  beforeEach(() => {
    mockPost.mockReset();
  });

  it("POSTs the key hash to /key/block when blocking", async () => {
    mockPost.mockResolvedValueOnce({ blocked: true });

    const result = await setKeyBlockedState("sk-access", { keyToken: "hashed-token", blocked: true });

    expect(mockPost).toHaveBeenCalledWith("/key/block", {
      accessToken: "sk-access",
      body: { key: "hashed-token" },
    });
    expect(result).toEqual({ blocked: true });
  });

  it("POSTs the key hash to /key/unblock when unblocking", async () => {
    mockPost.mockResolvedValueOnce({ blocked: false });

    const result = await setKeyBlockedState("sk-access", { keyToken: "hashed-token", blocked: false });

    expect(mockPost).toHaveBeenCalledWith("/key/unblock", {
      accessToken: "sk-access",
      body: { key: "hashed-token" },
    });
    expect(result).toEqual({ blocked: false });
  });

  it("falls back to the requested state when the response has no blocked field", async () => {
    mockPost.mockResolvedValueOnce(null);

    const result = await setKeyBlockedState("sk-access", { keyToken: "hashed-token", blocked: true });

    expect(result).toEqual({ blocked: true });
  });
});

describe("useSetKeyBlockedState", () => {
  beforeEach(() => {
    mockPost.mockReset();
    mockUseAuthorized.mockReturnValue({ accessToken: "sk-access" });
  });

  it("invalidates key queries after a successful mutation", async () => {
    mockPost.mockResolvedValueOnce({ blocked: true });
    const { queryClient, wrapper } = createWrapper();
    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries");

    const { result } = renderHook(() => useSetKeyBlockedState(), { wrapper });
    result.current.mutate({ keyToken: "hashed-token", blocked: true });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: ["keys"] });
  });

  it("surfaces request failures as mutation errors", async () => {
    mockPost.mockRejectedValueOnce(new Error("Key not found."));
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useSetKeyBlockedState(), { wrapper });
    result.current.mutate({ keyToken: "missing", blocked: true });

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(result.current.error?.message).toBe("Key not found.");
  });

  it("errors without an access token", async () => {
    mockUseAuthorized.mockReturnValue({ accessToken: null });
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useSetKeyBlockedState(), { wrapper });
    result.current.mutate({ keyToken: "hashed-token", blocked: true });

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(mockPost).not.toHaveBeenCalled();
  });
});
