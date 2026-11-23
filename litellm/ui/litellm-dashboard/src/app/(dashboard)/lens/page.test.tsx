import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import LensPage from "./page";

const { auth, workspace } = vi.hoisted(() => ({ auth: vi.fn(), workspace: vi.fn() }));
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: auth }));
vi.mock("@/components/lens/EmbeddedLens", () => ({
  EmbeddedLens: () => {
    workspace();
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

  it("shows the Lens page once the dashboard is signed in", () => {
    auth.mockReturnValue({ accessToken: "test-token", userRole: "Admin", isViewOnly: false });
    render(<LensPage />);
    expect(screen.getByText("Lens workspace")).toBeVisible();
  });
});
