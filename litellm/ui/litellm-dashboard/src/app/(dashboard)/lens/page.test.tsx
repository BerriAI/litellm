import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import LensPage from "./page";

const { auth, workspace } = vi.hoisted(() => ({
  auth: vi.fn<() => { token: string | null; authLoading: boolean }>(),
  workspace: vi.fn(),
}));
vi.mock("@/contexts/AuthContext", () => ({ useAuth: auth }));
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
    auth.mockReturnValue({ token: session(), authLoading: false });
  });

  it.each([
    { token: null, authLoading: false },
    { token: "invalid-jwt", authLoading: false },
    { token: session(), authLoading: true },
    { token: session({ exp: 1 }), authLoading: false },
  ])("does not mount Lens before the mounted session is valid %j", (state) => {
    auth.mockReturnValue(state);
    render(<LensPage />);
    expect(screen.queryByText("Lens workspace")).not.toBeInTheDocument();
    expect(workspace).not.toHaveBeenCalled();
  });

  it.each([undefined, 42, "", "   "])("does not mount Lens with an invalid access key %j", (key) => {
    auth.mockReturnValue({ token: session({ key }), authLoading: false });
    render(<LensPage />);
    expect(screen.queryByText("Lens workspace")).not.toBeInTheDocument();
    expect(workspace).not.toHaveBeenCalled();
  });

  it("passes the key and header from the same mounted session to Lens", () => {
    render(<LensPage />);
    expect(screen.getByText("Lens workspace")).toBeVisible();
    expect(workspace).toHaveBeenCalledWith({ accessToken: "dashboard-key", headerName: "X-Gateway-Key" });
  });

  it.each([undefined, 42, "", "   "])("defaults an invalid header %j to Authorization", (auth_header_name) => {
    auth.mockReturnValue({ token: session({ auth_header_name }), authLoading: false });
    render(<LensPage />);
    expect(screen.getByText("Lens workspace")).toBeVisible();
    expect(workspace).toHaveBeenCalledWith({ accessToken: "dashboard-key", headerName: "Authorization" });
  });
});
