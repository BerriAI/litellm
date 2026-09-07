/* @vitest-environment jsdom */
import { render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import AutoRoutersTabPanel from "./AutoRoutersTabPanel";

const panelProps = vi.fn();
vi.mock("../components/AutoRouters/AutoRoutersPanel", () => ({
  AutoRoutersPanel: (props: Record<string, unknown>) => {
    panelProps(props);
    return <div data-testid="auto-routers-panel" />;
  },
}));

const mockUseAuthorized = vi.fn();
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: () => mockUseAuthorized() }));
vi.mock("@/app/(dashboard)/hooks/teams/useTeams", () => ({ useTeams: () => ({ data: [] }) }));
vi.mock("@/app/(dashboard)/hooks/uiSettings/useUISettings", () => ({
  useUISettings: () => ({ data: { values: {} } }),
}));

const SESSION = { accessToken: "at", userRole: "Admin", userId: "u1", isViewOnly: false };

const lastProps = () => panelProps.mock.calls.at(-1)?.[0] as { createScope: string };

describe("AutoRoutersTabPanel", () => {
  it("grants an unscoped create to a real proxy admin", () => {
    mockUseAuthorized.mockReturnValue(SESSION);
    render(<AutoRoutersTabPanel />);
    expect(lastProps().createScope).toBe("unscoped-ok");
  });

  // The masqueraded "Admin" a proxy_admin_viewer session carries: POST /model/new 403s it,
  // so the panel must not be told it may create.
  it("withholds the create affordance from a view-only admin session", () => {
    mockUseAuthorized.mockReturnValue({ ...SESSION, isViewOnly: true });
    render(<AutoRoutersTabPanel />);
    expect(lastProps().createScope).toBe("forbidden");
  });
});
