/* @vitest-environment jsdom */
import { render } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

const { mockPush } = vi.hoisted(() => ({ mockPush: vi.fn() }));
vi.mock("next/navigation", () => ({ useRouter: () => ({ push: mockPush }) }));
vi.mock("@/components/networking", () => ({ serverRootPath: "" }));

import { createTabRoutes } from "@/utils/tabRoutes";
import { TabRouteBar } from "./TabRouteBar";

const routes = createTabRoutes("caching", ["health", "settings"] as const);
const TABS = [
  { key: "analytics", label: "Cache Analytics" },
  { key: "health", label: "Cache Health" },
  { key: "settings", label: "Cache Settings" },
];

const renderBar = (activeKey = "analytics") =>
  render(<TabRouteBar routes={routes} baseTabKey="analytics" activeKey={activeKey} tabs={TABS} />);

describe("TabRouteBar", () => {
  beforeEach(() => {
    mockPush.mockClear();
  });

  it("renders each tab as an anchor with its trailing-slash href (base tab maps to the root)", () => {
    const { getByRole } = renderBar();
    expect(getByRole("tab", { name: "Cache Analytics" })).toHaveAttribute("href", "/ui/caching/");
    expect(getByRole("tab", { name: "Cache Health" })).toHaveAttribute("href", "/ui/caching/health/");
    expect(getByRole("tab", { name: "Cache Settings" })).toHaveAttribute("href", "/ui/caching/settings/");
  });

  it("marks the active tab selected from activeKey", () => {
    const { getByRole } = renderBar("health");
    expect(getByRole("tab", { name: "Cache Health" })).toHaveAttribute("aria-selected", "true");
    expect(getByRole("tab", { name: "Cache Analytics" })).toHaveAttribute("aria-selected", "false");
  });

  it("soft-navigates on a plain left click (preventing the full-page anchor load)", async () => {
    const user = userEvent.setup();
    const { getByRole } = renderBar();
    await user.click(getByRole("tab", { name: "Cache Health" }));
    expect(mockPush).toHaveBeenCalledWith("/ui/caching/health/");
  });

  it("lets the browser handle a modifier-click so open-in-new-tab works", async () => {
    const user = userEvent.setup();
    const { getByRole } = renderBar();
    await user.keyboard("[ControlLeft>]");
    await user.click(getByRole("tab", { name: "Cache Health" }));
    await user.keyboard("[/ControlLeft]");
    expect(mockPush).not.toHaveBeenCalled();
  });
});
