/* @vitest-environment jsdom */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useModelDeprecations } from "./useModelDeprecations";

const authState = vi.hoisted(() => ({ accessToken: "token", userRole: "internal_user" }));
const emptyDeprecationsResponse = {
  checked_at: "2026-11-04T00:00:00Z",
  warn_within_days: 30,
  deprecated: [],
  imminent: [],
  upcoming: [],
};
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({ accessToken: authState.accessToken, userRole: authState.userRole }),
}));

describe("useModelDeprecations", () => {
  let fetchMock: ReturnType<typeof vi.fn<typeof fetch>>;

  beforeEach(() => {
    authState.accessToken = "token";
    authState.userRole = "internal_user";
    fetchMock = vi.fn<typeof fetch>().mockResolvedValue(Response.json(emptyDeprecationsResponse));
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("does not fetch deprecations for an internal user", () => {
    const queryClient = new QueryClient();
    const wrapper = ({ children }: { children: ReactNode }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );

    renderHook(() => useModelDeprecations(), { wrapper });

    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("fetches deprecations for a proxy admin viewer", async () => {
    authState.userRole = "proxy_admin_viewer";
    const queryClient = new QueryClient();
    const wrapper = ({ children }: { children: ReactNode }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );

    renderHook(() => useModelDeprecations(), { wrapper });

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    const request = fetchMock.mock.calls[0][0];
    expect(new URL(request instanceof Request ? request.url : String(request), "http://localhost").pathname).toBe(
      "/model/deprecations",
    );
  });
});
