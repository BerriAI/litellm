import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import LensPage from "./page";

const { auth, workspace } = vi.hoisted(() => ({ auth: vi.fn(), workspace: vi.fn() }));
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: auth }));
vi.mock("@/components/lens/LensWorkspace", () => ({
  LensWorkspace: (props: { accessToken: string; userRole: string; readOnly: boolean }) => {
    workspace(props);
    return <div>Lens workspace</div>;
  },
}));

describe("Lens route", () => {
  beforeEach(() => vi.clearAllMocks());

  it("waits for an access token", () => {
    auth.mockReturnValue({ accessToken: null, userRole: "Admin", isViewOnly: false });
    render(<LensPage />);
    expect(screen.queryByText("Lens workspace")).not.toBeInTheDocument();
    expect(workspace).not.toHaveBeenCalled();
  });

  it.each([
    { userRole: "Admin", isViewOnly: false },
    { userRole: "Admin Viewer", isViewOnly: true },
    { userRole: null, isViewOnly: false },
  ])("passes authorization to the workspace for $userRole", (session) => {
    auth.mockReturnValue({ accessToken: "test-token", ...session });
    render(<LensPage />);
    expect(screen.getByText("Lens workspace")).toBeVisible();
    expect(workspace).toHaveBeenCalledWith({
      accessToken: "test-token",
      userRole: session.userRole ?? "",
      readOnly: session.isViewOnly,
    });
  });
});
