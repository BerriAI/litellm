/* @vitest-environment jsdom */
import React from "react";
import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import useAuthorized from "./useAuthorized";

vi.unmock("@/app/(dashboard)/hooks/useAuthorized");

const replaceMock = vi.fn();

const uiConfigResponse = {
  server_root_path: "/llmproxy",
  proxy_base_url: null,
  auto_redirect_to_sso: false,
  admin_ui_disabled: false,
  sso_configured: false,
};

let uiConfigGate = Promise.withResolvers<typeof uiConfigResponse>();

const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
  const url = typeof input === "string" ? input : input.toString();
  if (url.endsWith("/ui/litellm-runtime-config.json")) {
    return { ok: false } as Response;
  }
  if (!url.endsWith("/.well-known/litellm-ui-config")) {
    throw new Error(`unexpected fetch: ${url}`);
  }
  const config = await uiConfigGate.promise;
  return { ok: true, json: async () => config } as unknown as Response;
});

const wrapper = ({ children }: { children: React.ReactNode }) => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  return React.createElement(QueryClientProvider, { client: queryClient }, children);
};

describe("useAuthorized under SERVER_ROOT_PATH", () => {
  const originalLocation = window.location;

  beforeEach(() => {
    uiConfigGate = Promise.withResolvers<typeof uiConfigResponse>();
    vi.stubGlobal("fetch", fetchMock);
    Object.defineProperty(window, "location", {
      value: {
        href: "http://proxy.example/llmproxy/ui/?page=virtual-keys",
        origin: "http://proxy.example",
        hostname: "proxy.example",
        pathname: "/llmproxy/ui/",
        search: "?page=virtual-keys",
        protocol: "http:",
        replace: replaceMock,
      },
      writable: true,
    });
    document.cookie = "token=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;";
  });

  afterEach(() => {
    Object.defineProperty(window, "location", { value: originalLocation, writable: true });
    vi.unstubAllGlobals();
    replaceMock.mockReset();
    fetchMock.mockClear();
  });

  it("sends an unauthenticated visitor to a login URL that keeps the server root path", async () => {
    renderHook(() => useAuthorized(), { wrapper });
    await act(async () => uiConfigGate.resolve(uiConfigResponse));

    await waitFor(() => {
      expect(replaceMock).toHaveBeenCalled();
    });

    expect(replaceMock).toHaveBeenCalledTimes(1);
    const { origin, pathname } = new URL(replaceMock.mock.calls[0][0] as string);
    expect(origin).toBe("http://proxy.example");
    expect(pathname).toBe("/llmproxy/ui/login/");
  });

  it("does not redirect before the UI config resolves the server root path", async () => {
    renderHook(() => useAuthorized(), { wrapper });

    expect(replaceMock).not.toHaveBeenCalled();
    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(expect.stringMatching(/\/\.well-known\/litellm-ui-config$/));
    });
    expect(replaceMock).not.toHaveBeenCalled();
    await act(async () => uiConfigGate.resolve(uiConfigResponse));

    await waitFor(() => {
      expect(replaceMock).toHaveBeenCalled();
    });
  });
});
