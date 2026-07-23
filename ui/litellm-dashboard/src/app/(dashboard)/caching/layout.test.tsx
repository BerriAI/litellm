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

  it("renders the tab bar and the active tab's page content", () => {
    const { getByRole, getByTestId } = renderLayout();
    expect(getByRole("tab", { name: "Cache Analytics" })).toBeInTheDocument();
    expect(getByRole("tab", { name: "Cache Health" })).toBeInTheDocument();
    expect(getByRole("tab", { name: "Cache Settings" })).toBeInTheDocument();
    expect(getByRole("tab", { name: "Coordination Redis" })).toBeInTheDocument();
    expect(getByTestId("tab-content")).toHaveTextContent("CHILD");
  });

  it("navigates to a tab's path when its tab is clicked", async () => {
    const { getByRole } = renderLayout();
    await act(async () => {
      getByRole("tab", { name: "Cache Health" }).click();
    });
    expect(mockPush).toHaveBeenCalledWith(expect.stringMatching(/\/caching\/health\/$/));
  });

  it("routes the base tab back to the caching root (no slug)", async () => {
    navState.pathname = "/caching/health";
    const { getByRole } = renderLayout();
    await act(async () => {
      getByRole("tab", { name: "Cache Analytics" }).click();
    });
    expect(mockPush).toHaveBeenCalledWith(expect.stringMatching(/\/caching\/$/));
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
