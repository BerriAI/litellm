/* @vitest-environment jsdom */
import { act, render } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import CachingLayout from "./layout";

const { mockPush, navState } = vi.hoisted(() => ({
  mockPush: vi.fn(),
  navState: { pathname: "/caching" },
}));
vi.mock("next/navigation", () => ({
  usePathname: () => navState.pathname,
  useRouter: () => ({ push: mockPush }),
}));

vi.mock("@/components/networking", () => ({ serverRootPath: "" }));

const renderLayout = () =>
  render(
    <CachingLayout>
      <div data-testid="tab-content">CHILD</div>
    </CachingLayout>,
  );

describe("CachingLayout", () => {
  beforeEach(() => {
    navState.pathname = "/caching";
    mockPush.mockClear();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (global as any).ResizeObserver = class {
      observe() {}
      unobserve() {}
      disconnect() {}
    };
  });

  it("renders the four tabs and the active tab's page content", () => {
    const { getByRole, getByTestId } = renderLayout();
    for (const name of ["Cache Analytics", "Cache Health", "Cache Settings", "Coordination Redis"]) {
      expect(getByRole("tab", { name })).toBeInTheDocument();
    }
    expect(getByTestId("tab-content")).toHaveTextContent("CHILD");
  });

  it("marks the base route's Cache Analytics tab active", () => {
    const { getByRole } = renderLayout();
    expect(getByRole("tab", { name: "Cache Analytics" })).toHaveAttribute("aria-selected", "true");
  });

  it("derives the active tab from a nested pathname", () => {
    navState.pathname = "/ui/caching/settings";
    const { getByRole } = renderLayout();
    expect(getByRole("tab", { name: "Cache Settings" })).toHaveAttribute("aria-selected", "true");
    expect(getByRole("tab", { name: "Cache Analytics" })).toHaveAttribute("aria-selected", "false");
  });

  it("redirects to the base caching path when the tab slug is unknown", async () => {
    const replaceMock = vi.fn();
    const originalLocation = window.location;
    Object.defineProperty(window, "location", {
      configurable: true,
      value: { replace: replaceMock, assign: vi.fn(), href: "http://localhost/", pathname: "/", search: "" },
    });
    navState.pathname = "/caching/bogus";
    await act(async () => {
      renderLayout();
    });
    expect(replaceMock).toHaveBeenCalledWith(expect.stringMatching(/\/caching\/$/));
    Object.defineProperty(window, "location", { configurable: true, value: originalLocation });
  });
});
