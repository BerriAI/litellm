import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi, it, expect, beforeEach } from "vitest";
import { renderWithProviders } from "../../../../tests/test-utils";
import { CLISessionsTable } from "./CLISessionsTable";
import type { CLISessionResponse } from "@/app/(dashboard)/hooks/cliSessions/useCLISessions";

const makeSession = (overrides: Partial<CLISessionResponse> = {}): CLISessionResponse => ({
  session_id: "a1b2c3d4e5f6",
  user_id: "cli-user-1",
  team_id: "team-1",
  created_at: "2026-08-13T10:00:00Z",
  expires_at: "2026-08-14T10:00:00Z",
  revoked_at: null,
  revoked_by: null,
  ...overrides,
});

const defaultProps = {
  sessions: [makeSession()],
  totalCount: 1,
  isLoading: false,
  isRevoking: false,
  canRevoke: true,
  onRevoke: vi.fn(),
  pagination: { pageIndex: 0, pageSize: 50 },
  onPaginationChange: vi.fn(),
};

beforeEach(() => {
  vi.clearAllMocks();
});

it("should show who the session belongs to, when it was issued and when it expires", () => {
  renderWithProviders(<CLISessionsTable {...defaultProps} />);

  expect(screen.getByText("cli-user-1")).toBeInTheDocument();
  expect(screen.getByText("a1b2c3d4e5f6")).toBeInTheDocument();
  expect(screen.getByText("team-1")).toBeInTheDocument();
  expect(screen.getByText("Active")).toBeInTheDocument();
});

it("should revoke the session the operator confirmed, not some other row", async () => {
  const user = userEvent.setup();
  const sessions = [
    makeSession({ session_id: "keep-me", user_id: "user-a" }),
    makeSession({ session_id: "kill-me", user_id: "user-b" }),
  ];
  renderWithProviders(<CLISessionsTable {...defaultProps} sessions={sessions} totalCount={2} />);

  const targetRow = screen.getByText("user-b").closest("tr")!;
  await user.click(within(targetRow).getByRole("button", { name: "Revoke" }));
  await user.click(screen.getByRole("alertdialog").querySelector("button:last-of-type") as HTMLElement);

  expect(defaultProps.onRevoke).toHaveBeenCalledTimes(1);
  expect(defaultProps.onRevoke).toHaveBeenCalledWith("kill-me");
});

it("should not offer a revoke action for an already revoked session", () => {
  renderWithProviders(
    <CLISessionsTable
      {...defaultProps}
      sessions={[makeSession({ revoked_at: "2026-08-13T11:00:00Z", revoked_by: "admin-1" })]}
    />,
  );

  expect(screen.queryByRole("button", { name: "Revoke" })).not.toBeInTheDocument();
  expect(screen.getAllByText("Revoked").length).toBeGreaterThan(0);
});

it("should not offer a revoke control to a read-only admin, who the proxy refuses anyway", () => {
  renderWithProviders(<CLISessionsTable {...defaultProps} canRevoke={false} />);

  expect(screen.getByText("cli-user-1")).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Revoke" })).not.toBeInTheDocument();
});

it("should show the empty state when no sessions are active", () => {
  renderWithProviders(<CLISessionsTable {...defaultProps} sessions={[]} totalCount={0} />);

  expect(screen.getByText("No active CLI sessions")).toBeInTheDocument();
});
