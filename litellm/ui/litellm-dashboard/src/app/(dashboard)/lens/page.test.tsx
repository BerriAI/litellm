import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import LensPage from "./page";

const { auth, session, workspace } = vi.hoisted(() => ({ auth: vi.fn(), session: vi.fn(), workspace: vi.fn() }));
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: auth }));
vi.mock("@/contexts/AuthContext", () => ({ useAuth: session }));
vi.mock("@/components/lens/EmbeddedLens", () => ({
  EmbeddedLens: ({ dashboardToken }: { dashboardToken: string | null }) => {
    workspace(dashboardToken);
    return <div>Lens workspace</div>;
  },
}));

describe("Lens route", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    session.mockReturnValue({ token: "mounted-dashboard-token" });
  });

  it("waits for an access token", () => {
    auth.mockReturnValue({ accessToken: null, userRole: "Admin", isViewOnly: false });
    render(<LensPage />);
    expect(screen.queryByText("Lens workspace")).not.toBeInTheDocument();
    expect(workspace).not.toHaveBeenCalled();
  });

  it("passes the mounted dashboard session to Lens after authorization", () => {
    auth.mockReturnValue({ token: "cookie-token", accessToken: "test-key" });
    render(<LensPage />);
    expect(screen.getByText("Lens workspace")).toBeVisible();
    expect(workspace).toHaveBeenCalledWith("mounted-dashboard-token");
  });
});
