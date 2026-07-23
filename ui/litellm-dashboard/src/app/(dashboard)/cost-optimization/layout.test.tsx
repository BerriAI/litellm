/* @vitest-environment jsdom */
import { act, render } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import CostOptimizationLayout from "./layout";

const { mockPush, navState } = vi.hoisted(() => ({
  mockPush: vi.fn(),
  navState: { pathname: "/cost-optimization" },
}));
vi.mock("next/navigation", () => ({
  usePathname: () => navState.pathname,
  useRouter: () => ({ push: mockPush }),
}));

vi.mock("@/components/networking", () => ({ serverRootPath: "" }));

const renderLayout = () =>
  render(
    <CostOptimizationLayout>
      <div data-testid="tab-content">CHILD</div>
    </CostOptimizationLayout>,
  );

describe("CostOptimizationLayout", () => {
  beforeEach(() => {
    navState.pathname = "/cost-optimization";
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
    expect(getByRole("tab", { name: "Usage" })).toBeInTheDocument();
    expect(getByRole("tab", { name: "Prompt Compression" })).toBeInTheDocument();
    expect(getByRole("tab", { name: "Autorouter" })).toBeInTheDocument();
    expect(getByRole("tab", { name: "Prompt Caching" })).toBeInTheDocument();
    expect(getByTestId("tab-content")).toHaveTextContent("CHILD");
  });

  it("marks the base route's Usage tab active", () => {
    const { getByRole } = renderLayout();
    expect(getByRole("tab", { name: "Usage" })).toHaveAttribute("aria-selected", "true");
    expect(getByRole("tab", { name: "Prompt Compression" })).toHaveAttribute("aria-selected", "false");
  });

  it("navigates to a tab's path when its tab is clicked", async () => {
    const { getByRole } = renderLayout();
    await act(async () => {
      getByRole("tab", { name: "Prompt Compression" }).click();
    });
    expect(mockPush).toHaveBeenCalledWith(expect.stringMatching(/\/cost-optimization\/compression\/$/));
  });

  it("routes the base tab back to the cost-optimization root (no slug)", async () => {
    navState.pathname = "/cost-optimization/autorouter";
    const { getByRole } = renderLayout();
    await act(async () => {
      getByRole("tab", { name: "Usage" }).click();
    });
    expect(mockPush).toHaveBeenCalledWith(expect.stringMatching(/\/cost-optimization\/$/));
  });

  it("redirects to the base cost-optimization path when the tab slug is unknown", async () => {
    const replaceMock = vi.fn();
    const originalLocation = window.location;
    Object.defineProperty(window, "location", {
      configurable: true,
      value: { replace: replaceMock, assign: vi.fn(), href: "http://localhost/", pathname: "/", search: "" },
    });
    navState.pathname = "/cost-optimization/bogus";
    await act(async () => {
      renderLayout();
    });
    expect(replaceMock).toHaveBeenCalledWith(expect.stringMatching(/\/cost-optimization\/$/));
    Object.defineProperty(window, "location", { configurable: true, value: originalLocation });
  });
});
