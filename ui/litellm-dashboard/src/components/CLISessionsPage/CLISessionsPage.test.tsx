/* @vitest-environment jsdom */
import { screen } from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../tests/test-utils";
import CLISessionsPage from "./CLISessionsPage";
import { effectiveSessionRole, isViewOnlySessionRole } from "@/utils/roles";

const session = {
  session_id: "a1b2c3d4e5f6",
  user_id: "cli-user-1",
  team_id: null,
  created_at: "2026-08-13T10:00:00Z",
  expires_at: "2026-08-14T10:00:00Z",
  revoked_at: null,
  revoked_by: null,
};

// The page reads identity straight off the session cookie, so the raw proxy role is
// what a test needs to vary; useAuthorized derives the rest exactly as it does live.
const rawUserRole = vi.fn(() => "proxy_admin");
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({
    accessToken: "sk-test",
    userRole: effectiveSessionRole(rawUserRole()),
    isViewOnly: isViewOnlySessionRole(rawUserRole()),
  }),
}));

vi.mock("@/app/(dashboard)/hooks/cliSessions/useCLISessions", () => ({
  useCLISessions: () => ({ data: { sessions: [session], total_count: 1 }, isLoading: false }),
  useRevokeCLISession: () => ({ mutate: vi.fn(), isPending: false }),
}));

beforeEach(() => {
  rawUserRole.mockReturnValue("proxy_admin");
});

it("should offer Revoke to a proxy admin", () => {
  renderWithProviders(<CLISessionsPage />);

  expect(screen.getByRole("button", { name: "Revoke" })).toBeInTheDocument();
});

it("should not offer Revoke to a read-only admin, whom the proxy refuses on the revoke route", () => {
  // effectiveSessionRole normalizes proxy_admin_viewer to "Admin" for read parity, so a
  // role-only check would hand this user a control that always comes back 403.
  rawUserRole.mockReturnValue("proxy_admin_viewer");

  renderWithProviders(<CLISessionsPage />);

  expect(screen.getByText("cli-user-1")).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Revoke" })).not.toBeInTheDocument();
});
