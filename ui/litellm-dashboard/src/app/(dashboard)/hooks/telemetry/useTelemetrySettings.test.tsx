import { afterEach, describe, expect, it, vi } from "vitest";
import { act, renderHook } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React, { ReactNode } from "react";
import { queryClient as appQueryClient } from "@/contexts/ReactQueryProvider";
import { backgroundFetchClient, fetchClient } from "@/lib/http/api";
import { registerErrorHandler } from "@/lib/http/runtime";
import { recordUiEvent } from "@/lib/telemetry/uiEvents";
import { type TelemetrySettings, useTelemetrySettings, useUpdateTelemetrySettings } from "./useTelemetrySettings";

const { authorized } = vi.hoisted(() => ({ authorized: { accessToken: "sk-test", userRole: "org_admin" } }));
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: () => authorized }));

const settingsWith = (pageNavigation: boolean): TelemetrySettings =>
  ({
    groups: [
      { group: "heartbeat", enabled: true, requires: null },
      { group: "page_navigation", enabled: pageNavigation, requires: "heartbeat" },
    ],
  }) as unknown as TelemetrySettings;

const wrapperFor = (client: QueryClient) =>
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
  };

describe("useTelemetrySettings", () => {
  afterEach(() => vi.restoreAllMocks());

  it.each([
    ["org_admin", 0],
    ["proxy_admin_viewer", 1],
  ])("only asks for the settings as a proxy admin tier role (%s)", async (role, calls) => {
    authorized.userRole = role;
    const get = vi
      .spyOn(backgroundFetchClient, "GET")
      .mockResolvedValue({ data: settingsWith(false), response: new Response() } as never);

    renderHook(() => useTelemetrySettings(), { wrapper: wrapperFor(new QueryClient()) });
    await act(() => new Promise((resolve) => setTimeout(resolve, 0)));

    expect(get).toHaveBeenCalledTimes(calls);
  });

  it("does not send a failed settings read through the session error handler", async () => {
    authorized.userRole = "proxy_admin";
    const handler = vi.fn();
    registerErrorHandler(handler);
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ detail: "forbidden" }), {
        status: 403,
        headers: { "content-type": "application/json" },
      }),
    );

    const { result } = renderHook(() => useTelemetrySettings(), {
      wrapper: wrapperFor(new QueryClient({ defaultOptions: { queries: { retry: false } } })),
    });
    await vi.waitFor(() => expect(result.current.isError).toBe(true));

    registerErrorHandler(() => {});
    expect(handler).not.toHaveBeenCalled();
  });
});

describe("useUpdateTelemetrySettings", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    appQueryClient.clear();
  });

  it("leaves page navigation to the proxy until its next window instead of flipping it on save", async () => {
    const get = vi
      .spyOn(backgroundFetchClient, "GET")
      .mockResolvedValue({ data: { enabled: false }, response: new Response() } as never);
    const post = vi
      .spyOn(backgroundFetchClient, "POST")
      .mockResolvedValue({ data: undefined, response: new Response() });
    vi.spyOn(fetchClient, "PUT").mockResolvedValue({ data: settingsWith(true), response: new Response() });
    await recordUiEvent({ page: "telemetry", action: "view" });
    const { result } = renderHook(() => useUpdateTelemetrySettings(), { wrapper: wrapperFor(new QueryClient()) });

    await act(() => result.current.mutateAsync(["heartbeat", "page_navigation"]));
    await recordUiEvent({ page: "telemetry", action: "click", target: "tab=ui" });

    expect(post).not.toHaveBeenCalled();
    expect(get).toHaveBeenCalledOnce();
  });
});
