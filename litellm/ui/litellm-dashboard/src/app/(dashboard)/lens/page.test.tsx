import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import LensPage from "./page";

const { auth, header, workspace } = vi.hoisted(() => ({ auth: vi.fn(), header: vi.fn(), workspace: vi.fn() }));
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: auth }));
vi.mock("@/components/networking", () => ({ getGlobalLitellmHeaderName: header }));
vi.mock("@/components/lens/EmbeddedLens", () => ({
  EmbeddedLens: (credentials: { accessToken: string; headerName: string }) => {
    workspace(credentials);
    return <div>Lens workspace</div>;
  },
}));

describe("Lens route", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    header.mockReturnValue("X-Gateway-Key");
    auth.mockReturnValue({ accessToken: "dashboard-key", isAuthorized: true });
  });

  it.each([
    { accessToken: null, isAuthorized: true },
    { accessToken: "expired-key", isAuthorized: false },
  ])("does not mount Lens without authorized credentials %j", (credentials) => {
    auth.mockReturnValue(credentials);
    render(<LensPage />);
    expect(screen.queryByText("Lens workspace")).not.toBeInTheDocument();
    expect(workspace).not.toHaveBeenCalled();
  });

  it("passes the dashboard access key and configured header to Lens", () => {
    render(<LensPage />);
    expect(screen.getByText("Lens workspace")).toBeVisible();
    expect(workspace).toHaveBeenCalledWith({ accessToken: "dashboard-key", headerName: "X-Gateway-Key" });
  });

  it("immediately removes Lens when the route guard rejects a previously decoded key", () => {
    const { rerender } = render(<LensPage />);
    expect(screen.getByText("Lens workspace")).toBeVisible();

    auth.mockReturnValue({ accessToken: "dashboard-key", isAuthorized: false });
    rerender(<LensPage />);

    expect(screen.queryByText("Lens workspace")).not.toBeInTheDocument();
    expect(workspace).toHaveBeenCalledTimes(1);
  });
});
