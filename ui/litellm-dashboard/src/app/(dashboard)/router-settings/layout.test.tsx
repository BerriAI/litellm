/* @vitest-environment jsdom */
import { act, render } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import RouterSettingsLayout from "./layout";

const { mockPush, navState } = vi.hoisted(() => ({
  mockPush: vi.fn(),
  navState: { pathname: "/router-settings" },
}));
vi.mock("next/navigation", () => ({
  usePathname: () => navState.pathname,
  useRouter: () => ({ push: mockPush }),
}));

vi.mock("@/components/networking", () => ({ serverRootPath: "" }));

const renderLayout = () =>
  render(
    <RouterSettingsLayout>
      <div data-testid="tab-content">CHILD</div>
    </RouterSettingsLayout>,
  );

describe("RouterSettingsLayout", () => {
  beforeEach(() => {
    navState.pathname = "/router-settings";
    mockPush.mockClear();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (global as any).ResizeObserver = class {
      observe() {}
      unobserve() {}
      disconnect() {}
    };
  });

  it("renders the tab bar and the active tab's page content", () => {
    const { getByRole, getByTestId } = renderLayout();
    for (const name of ["Loadbalancing", "Routing Groups", "Fallbacks", "Prompt Caching", "General"]) {
      expect(getByRole("tab", { name })).toBeInTheDocument();
    }
    expect(getByTestId("tab-content")).toHaveTextContent("CHILD");
  });

  it("marks the base route's Loadbalancing tab active", () => {
    const { getByRole } = renderLayout();
    expect(getByRole("tab", { name: "Loadbalancing" })).toHaveAttribute("aria-selected", "true");
    expect(getByRole("tab", { name: "General" })).toHaveAttribute("aria-selected", "false");
  });

  it("navigates to a tab's path when its tab is clicked", async () => {
    const { getByRole } = renderLayout();
    await act(async () => {
      getByRole("tab", { name: "Prompt Caching" }).click();
    });
    expect(mockPush).toHaveBeenCalledWith(expect.stringMatching(/\/router-settings\/prompt-caching\/$/));
  });

  it("routes the base tab back to the router-settings root (no slug)", async () => {
    navState.pathname = "/router-settings/general";
    const { getByRole } = renderLayout();
    await act(async () => {
      getByRole("tab", { name: "Loadbalancing" }).click();
    });
    expect(mockPush).toHaveBeenCalledWith(expect.stringMatching(/\/router-settings\/$/));
  });

  it("redirects to the base router-settings path when the tab slug is unknown", async () => {
    const replaceMock = vi.fn();
    const originalLocation = window.location;
    Object.defineProperty(window, "location", {
      configurable: true,
      value: { replace: replaceMock, assign: vi.fn(), href: "http://localhost/", pathname: "/", search: "" },
    });
    navState.pathname = "/router-settings/bogus";
    await act(async () => {
      renderLayout();
    });
    expect(replaceMock).toHaveBeenCalledWith(expect.stringMatching(/\/router-settings\/$/));
    Object.defineProperty(window, "location", { configurable: true, value: originalLocation });
  });
});
