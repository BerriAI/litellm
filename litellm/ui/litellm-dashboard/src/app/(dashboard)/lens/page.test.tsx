import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { getGlobalLitellmHeaderName, setGlobalLitellmHeaderName } from "@/components/networking";
import LensPage from "./page";

const { auth, workspace } = vi.hoisted(() => ({
  auth: vi.fn<() => { token: string | null; isLoading: boolean; isAuthorized: boolean }>(),
  workspace: vi.fn(),
}));
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: auth }));
vi.mock("@/components/lens/EmbeddedLens", () => ({
  EmbeddedLens: (credentials: { accessToken: string; headerName: string }) => {
    workspace(credentials);
    return <div>Lens workspace</div>;
  },
}));

const session = (claims: Record<string, unknown> = {}) =>
  `header.${btoa(JSON.stringify({ key: "dashboard-key", auth_header_name: "X-Gateway-Key", ...claims }))}.signature`;

describe("Lens route", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setGlobalLitellmHeaderName("Authorization");
    auth.mockReturnValue({ token: session(), isLoading: false, isAuthorized: true });
  });

  it.each([
    { token: session(), isLoading: true, isAuthorized: true },
    { token: null, isLoading: false, isAuthorized: false },
  ])("does not mount Lens until the dashboard session is loaded and authorized %j", (state) => {
    auth.mockReturnValue(state);
    render(<LensPage />);
    expect(screen.queryByText("Lens workspace")).not.toBeInTheDocument();
    expect(workspace).not.toHaveBeenCalled();
  });

  it("uses the authorized token key and custom header while the global header is still Authorization", () => {
    expect(getGlobalLitellmHeaderName()).toBe("Authorization");
    render(<LensPage />);
    expect(screen.getByText("Lens workspace")).toBeVisible();
    expect(workspace).toHaveBeenCalledWith({ accessToken: "dashboard-key", headerName: "X-Gateway-Key" });
  });

  it("uses Authorization when the dashboard session has no custom header", () => {
    auth.mockReturnValue({
      token: session({ auth_header_name: undefined }),
      isLoading: false,
      isAuthorized: true,
    });
    render(<LensPage />);
    expect(screen.getByText("Lens workspace")).toBeVisible();
    expect(workspace).toHaveBeenCalledWith({ accessToken: "dashboard-key", headerName: "Authorization" });
  });
});
