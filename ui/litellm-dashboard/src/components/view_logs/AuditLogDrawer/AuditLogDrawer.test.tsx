import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { AuditLogDrawer } from "./AuditLogDrawer";
import { AuditLogEntry } from "../columns";

vi.mock("../../common_components/DefaultProxyAdminTag", () => ({
  default: ({ userId }: { userId: string }) => <span data-testid="proxy-admin-tag">{userId}</span>,
}));

const mockLog: AuditLogEntry = {
  id: "log-1",
  updated_at: "2026-03-15T10:30:00Z",
  changed_by: "admin-user",
  changed_by_api_key: "sk-abc123hash",
  action: "created",
  table_name: "LiteLLM_TeamTable",
  object_id: "team-456",
  before_value: {},
  updated_values: { team_alias: "My Team", max_budget: 100 },
};

describe("AuditLogDrawer", () => {
  const defaultProps = {
    open: true,
    onClose: vi.fn(),
    log: mockLog,
  };

  it("should render", () => {
    render(<AuditLogDrawer {...defaultProps} />);
    expect(screen.getByText("created")).toBeInTheDocument();
  });

  it("should return null when log is null", () => {
    const { container } = render(<AuditLogDrawer open={true} onClose={vi.fn()} log={null} />);
    expect(container.innerHTML).toBe("");
  });

  it("should display the friendly table name", () => {
    render(<AuditLogDrawer {...defaultProps} />);
    expect(screen.getByText("Teams")).toBeInTheDocument();
  });

  it("should display the object ID", () => {
    render(<AuditLogDrawer {...defaultProps} />);
    expect(screen.getByText("team-456")).toBeInTheDocument();
  });

  it("should display the changed_by user", () => {
    render(<AuditLogDrawer {...defaultProps} />);
    expect(screen.getByText("admin-user")).toBeInTheDocument();
  });

  it("should display the API key hash", () => {
    render(<AuditLogDrawer {...defaultProps} />);
    expect(screen.getByText("sk-abc123hash")).toBeInTheDocument();
  });

  it("should show a close button", async () => {
    const onClose = vi.fn();
    const user = userEvent.setup();
    render(<AuditLogDrawer {...defaultProps} onClose={onClose} />);

    await user.click(screen.getByRole("button", { name: /close/i }));
    expect(onClose).toHaveBeenCalled();
  });

  it("should display raw table name when no friendly mapping exists", () => {
    const customLog = { ...mockLog, table_name: "CustomTable" };
    render(<AuditLogDrawer {...defaultProps} log={customLog} />);
    expect(screen.getByText("CustomTable")).toBeInTheDocument();
  });

  it("should show diff sections for updated action with changed fields", () => {
    const updatedLog: AuditLogEntry = {
      ...mockLog,
      action: "updated",
      before_value: { team_alias: "Old Name", max_budget: 50 },
      updated_values: { team_alias: "New Name", max_budget: 100 },
    };
    render(<AuditLogDrawer {...defaultProps} log={updatedLog} />);
    expect(screen.getByText("Before")).toBeInTheDocument();
    expect(screen.getByText("After")).toBeInTheDocument();
  });
});
