/* @vitest-environment jsdom */
import { act, render } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import LogsLayout from "./layout";

const { mockPush, navState } = vi.hoisted(() => ({
  mockPush: vi.fn(),
  navState: { pathname: "/logs" },
}));
vi.mock("next/navigation", () => ({
  usePathname: () => navState.pathname,
  useRouter: () => ({ push: mockPush }),
}));

vi.mock("@/components/networking", () => ({ serverRootPath: "" }));

const mockUseAuthorized = vi.fn();
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: () => mockUseAuthorized() }));

const READY = { accessToken: "at", token: "tok", userRole: "Admin", userId: "u1", premiumUser: false };

const renderLayout = () =>
  render(
    <LogsLayout>
      <div data-testid="tab-content">CHILD</div>
    </LogsLayout>,
  );

describe("LogsLayout", () => {
  beforeEach(() => {
    navState.pathname = "/logs";
    mockPush.mockClear();
    mockUseAuthorized.mockReturnValue(READY);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (global as any).ResizeObserver = class {
      observe() {}
      unobserve() {}
      disconnect() {}
    };
  });

  it("renders the four log tabs and the active tab's page content", () => {
    const { getByRole, getByTestId } = renderLayout();
    for (const name of ["Request Logs", "Audit Logs", "Deleted Keys", "Deleted Teams"]) {
      expect(getByRole("tab", { name })).toBeInTheDocument();
    }
    expect(getByTestId("tab-content")).toHaveTextContent("CHILD");
  });

  it("marks the base route's Request Logs tab active", () => {
    const { getByRole } = renderLayout();
    expect(getByRole("tab", { name: "Request Logs" })).toHaveAttribute("aria-selected", "true");
    expect(getByRole("tab", { name: "Audit Logs" })).toHaveAttribute("aria-selected", "false");
  });

  it("navigates to a tab's path when its tab is clicked", async () => {
    const { getByRole } = renderLayout();
    await act(async () => {
      getByRole("tab", { name: "Audit Logs" }).click();
    });
    expect(mockPush).toHaveBeenCalledWith(expect.stringMatching(/\/logs\/audit\/$/));
  });

  it("routes the base tab back to the logs root (no slug)", async () => {
    navState.pathname = "/logs/audit";
    const { getByRole } = renderLayout();
    await act(async () => {
      getByRole("tab", { name: "Request Logs" }).click();
    });
    expect(mockPush).toHaveBeenCalledWith(expect.stringMatching(/\/logs\/$/));
  });

  it("redirects to the base logs path when the tab slug is unknown", async () => {
    const replaceMock = vi.fn();
    const originalLocation = window.location;
    Object.defineProperty(window, "location", {
      configurable: true,
      value: { replace: replaceMock, assign: vi.fn(), href: "http://localhost/", pathname: "/", search: "" },
    });
    navState.pathname = "/logs/bogus";
    await act(async () => {
      renderLayout();
    });
    expect(replaceMock).toHaveBeenCalledWith(expect.stringMatching(/\/logs\/$/));
    Object.defineProperty(window, "location", { configurable: true, value: originalLocation });
  });

  it("shows a loading spinner and no tabs until credentials resolve", () => {
    mockUseAuthorized.mockReturnValue({ ...READY, accessToken: null });
    const { container, queryByRole } = renderLayout();
    expect(container.querySelector(".ant-spin")).toBeInTheDocument();
    expect(queryByRole("tab", { name: "Request Logs" })).not.toBeInTheDocument();
  });
});
