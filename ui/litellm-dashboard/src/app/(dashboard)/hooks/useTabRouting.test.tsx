/* @vitest-environment jsdom */
import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const { mockPush, navState } = vi.hoisted(() => ({
  mockPush: vi.fn(),
  navState: { pathname: "/logs" },
}));
vi.mock("next/navigation", () => ({
  usePathname: () => navState.pathname,
  useRouter: () => ({ push: mockPush }),
}));

vi.mock("@/components/networking", () => ({ serverRootPath: "" }));

import { createTabRoutes } from "@/utils/tabRoutes";
import { useTabRouting } from "./useTabRouting";

const routes = createTabRoutes("logs", ["audit", "deleted-keys", "deleted-teams"] as const);

const render = (ready = true) => {
  const config = {
    routes,
    baseTabKey: "request-logs",
    visibleKeys: ["audit", "deleted-keys", "deleted-teams"],
    ready,
  };
  return renderHook(() => useTabRouting(config));
};

describe("useTabRouting", () => {
  beforeEach(() => {
    navState.pathname = "/logs";
    mockPush.mockClear();
  });

  it("maps the base path to the base tab key", () => {
    const { result } = render();
    expect(result.current.activeSlug).toBe("");
    expect(result.current.activeKey).toBe("request-logs");
  });

  it("uses the slug itself as the active key for a known nested tab", () => {
    navState.pathname = "/ui/logs/audit";
    const { result } = render();
    expect(result.current.activeKey).toBe("audit");
  });

  it("falls back to the base tab key for an unknown slug", () => {
    navState.pathname = "/ui/logs/bogus";
    const { result } = render();
    expect(result.current.activeKey).toBe("request-logs");
  });

  it("redirects an unknown slug to the base href once ready", () => {
    const replaceMock = vi.fn();
    const originalLocation = window.location;
    Object.defineProperty(window, "location", { configurable: true, value: { replace: replaceMock } });
    navState.pathname = "/ui/logs/bogus";
    render(true);
    expect(replaceMock).toHaveBeenCalledWith("/ui/logs/");
    Object.defineProperty(window, "location", { configurable: true, value: originalLocation });
  });

  it("does not redirect while not ready (role/creds still loading)", () => {
    const replaceMock = vi.fn();
    const originalLocation = window.location;
    Object.defineProperty(window, "location", { configurable: true, value: { replace: replaceMock } });
    navState.pathname = "/ui/logs/bogus";
    render(false);
    expect(replaceMock).not.toHaveBeenCalled();
    Object.defineProperty(window, "location", { configurable: true, value: originalLocation });
  });

  it("pushes the tab href on change, mapping the base key back to the empty slug", () => {
    const { result } = render();
    result.current.onTabChange("audit");
    expect(mockPush).toHaveBeenCalledWith("/ui/logs/audit/");
    result.current.onTabChange("request-logs");
    expect(mockPush).toHaveBeenCalledWith("/ui/logs/");
  });
});
