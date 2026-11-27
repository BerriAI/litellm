/* @vitest-environment jsdom */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const replaceMock = vi.fn();

const stubPage = (pathname: string) => {
  Object.defineProperty(window, "location", {
    value: {
      href: `http://proxy.example${pathname}`,
      origin: "http://proxy.example",
      hostname: "proxy.example",
      pathname,
      search: "",
      protocol: "http:",
      replace: replaceMock,
    },
    writable: true,
  });
};

const stubFetch = (uiConfig: object | null) =>
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      if (uiConfig === null || String(input).endsWith("/litellm-runtime-config.json")) {
        throw new Error("network down");
      }
      return { ok: true, json: async () => uiConfig } as unknown as Response;
    }),
  );

const loginUrlAfterRedirect = async (loadUiConfig: boolean) => {
  vi.resetModules();
  const { getUiConfig } = await import("@/components/networking");
  const { redirectToLogin } = await import("./redirectToLogin");
  if (loadUiConfig) await getUiConfig();
  redirectToLogin();
  expect(replaceMock).toHaveBeenCalledTimes(1);
  return new URL(replaceMock.mock.calls[0][0] as string);
};

describe("redirectToLogin", () => {
  const originalLocation = window.location;

  beforeEach(() => {
    document.cookie = "litellm_return_url=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;";
  });

  afterEach(() => {
    Object.defineProperty(window, "location", { value: originalLocation, writable: true });
    vi.unstubAllGlobals();
    replaceMock.mockReset();
  });

  it("keeps the server root path when the UI config could not be loaded", async () => {
    stubPage("/9000/ui/");
    stubFetch(null);

    const loginUrl = await loginUrlAfterRedirect(false);

    expect(loginUrl.origin).toBe("http://proxy.example");
    expect(loginUrl.pathname).toBe("/9000/ui/login/");
  });

  it("does not repeat the root path when the proxy base URL already contains it", async () => {
    stubPage("/9000/ui/");
    stubFetch({ server_root_path: "/9000", proxy_base_url: "http://proxy.example/9000" });

    const loginUrl = await loginUrlAfterRedirect(true);

    expect(loginUrl.pathname).toBe("/9000/ui/login/");
  });

  it("sends the visitor to /ui/login/ when the proxy is mounted at the root", async () => {
    stubPage("/ui/");
    stubFetch({ server_root_path: "/", proxy_base_url: null });

    const loginUrl = await loginUrlAfterRedirect(true);

    expect(loginUrl.pathname).toBe("/ui/login/");
  });
});
