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

  it("renders the four tabs and the active tab's page content", () => {
    const { getByRole, getByTestId } = renderLayout();
    for (const name of ["Usage", "Prompt Compression", "Autorouter", "Prompt Caching"]) {
      expect(getByRole("tab", { name })).toBeInTheDocument();
    }
    expect(getByTestId("tab-content")).toHaveTextContent("CHILD");
  });

  it("marks the base route's Usage tab active", () => {
    const { getByRole } = renderLayout();
    expect(getByRole("tab", { name: "Usage" })).toHaveAttribute("aria-selected", "true");
  });

  it("derives the active tab from a nested pathname", () => {
    navState.pathname = "/ui/cost-optimization/compression";
    const { getByRole } = renderLayout();
    expect(getByRole("tab", { name: "Prompt Compression" })).toHaveAttribute("aria-selected", "true");
    expect(getByRole("tab", { name: "Usage" })).toHaveAttribute("aria-selected", "false");
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
