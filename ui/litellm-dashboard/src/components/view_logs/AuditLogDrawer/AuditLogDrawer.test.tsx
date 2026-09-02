import { describe, it, expect, vi, beforeEach } from "vitest";
import userEvent from "@testing-library/user-event";
import { render, screen, waitFor, within } from "@testing-library/react";
import moment from "moment";
import { AuditLogDrawer } from "./AuditLogDrawer";
import { AuditLogEntry } from "../AuditLogsTableColumns";

vi.mock("../../common_components/DefaultProxyAdminTag", () => ({
  default: ({ userId }: { userId: string }) => <span>{userId}</span>,
}));

const baseLog: AuditLogEntry = {
  id: "audit-1",
  updated_at: "2026-07-20T10:30:00Z",
  changed_by: "user-1",
  changed_by_api_key: "hashed-key-abc",
  action: "updated",
  table_name: "LiteLLM_TeamTable",
  object_id: "team-42",
  before_value: { max_budget: 10, tpm_limit: 100 },
  updated_values: { max_budget: 25, tpm_limit: 100 },
};

const defaultProps = { open: true, onClose: vi.fn(), log: baseLog };

function blockNamed(label: string) {
  const heading = screen.getByText(label);
  const block = heading.closest("div")?.parentElement;
  if (!block) throw new Error(`no block for ${label}`);
  return block as HTMLElement;
}

describe("AuditLogDrawer", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render nothing when there is no log", () => {
    const { container } = render(<AuditLogDrawer {...defaultProps} log={null} />);
    expect(container).toBeEmptyDOMElement();
    expect(screen.queryByText("Details")).not.toBeInTheDocument();
  });

  it("should show the action and the local timestamp in the header", () => {
    render(<AuditLogDrawer {...defaultProps} />);
    expect(screen.getByText("updated")).toBeInTheDocument();
    expect(screen.getByText(moment.utc(baseLog.updated_at).local().format("MMM D, YYYY HH:mm:ss"))).toBeInTheDocument();
  });

  it("should show the friendly table name for a known table", () => {
    render(<AuditLogDrawer {...defaultProps} />);
    expect(screen.getByText("Table")).toBeInTheDocument();
    expect(screen.getByText("Teams")).toBeInTheDocument();
  });

  it("should fall back to the raw table name when it is not mapped", () => {
    render(<AuditLogDrawer {...defaultProps} log={{ ...baseLog, table_name: "LiteLLM_SomethingElse" }} />);
    expect(screen.getByText("LiteLLM_SomethingElse")).toBeInTheDocument();
  });

  it("should show the object id, the actor and the api key hash", () => {
    render(<AuditLogDrawer {...defaultProps} />);
    expect(screen.getByText("team-42")).toBeInTheDocument();
    expect(screen.getByText("user-1")).toBeInTheDocument();
    expect(screen.getByText("hashed-key-abc")).toBeInTheDocument();
  });

  it("should show a placeholder when the log has no api key hash", () => {
    render(<AuditLogDrawer {...defaultProps} log={{ ...baseLog, changed_by_api_key: "" }} />);
    expect(screen.getByText("API Key (Hash)")).toBeInTheDocument();
    expect(screen.queryByText("hashed-key-abc")).not.toBeInTheDocument();
  });

  it("should call onClose when the close control is used", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    render(<AuditLogDrawer {...defaultProps} onClose={onClose} />);
    await user.click(screen.getByRole("button", { name: /close/i }));
    expect(onClose).toHaveBeenCalledOnce();
  });

  it("should show only the fields that changed in the before and after blocks", () => {
    render(<AuditLogDrawer {...defaultProps} />);

    expect(within(blockNamed("Before")).getByText(/"max_budget": 10/)).toBeInTheDocument();
    expect(within(blockNamed("After")).getByText(/"max_budget": 25/)).toBeInTheDocument();
    expect(screen.queryByText(/tpm_limit/)).not.toBeInTheDocument();
  });

  it("should note when an update has no differing fields", () => {
    render(<AuditLogDrawer {...defaultProps} log={{ ...baseLog, before_value: { a: 1 }, updated_values: { a: 1 } }} />);
    expect(screen.getAllByText(/No differing fields detected/).length).toBeGreaterThan(0);
  });

  it("should show N/A for a side with no values on a create", () => {
    render(
      <AuditLogDrawer
        {...defaultProps}
        log={{ ...baseLog, action: "created", before_value: {}, updated_values: { team_alias: "new team" } }}
      />,
    );
    expect(within(blockNamed("Before")).getByText("N/A")).toBeInTheDocument();
    expect(within(blockNamed("After")).getByText(/"team_alias": "new team"/)).toBeInTheDocument();
  });

  it("should render key-table updates as labelled plain text rather than json", () => {
    render(
      <AuditLogDrawer
        {...defaultProps}
        log={{
          ...baseLog,
          table_name: "LiteLLM_VerificationToken",
          before_value: { spend: 1, max_budget: 10 },
          updated_values: { spend: 2, max_budget: 10 },
        }}
      />,
    );
    expect(within(blockNamed("Before")).getByText("$1.000000")).toBeInTheDocument();
    expect(within(blockNamed("After")).getByText("$2.000000")).toBeInTheDocument();
    expect(screen.queryByText(/"spend"/)).not.toBeInTheDocument();
  });

  it("should copy the json of a block to the clipboard", async () => {
    const user = userEvent.setup();
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", { value: { writeText }, configurable: true });
    Object.defineProperty(window, "isSecureContext", { value: true, configurable: true });

    render(<AuditLogDrawer {...defaultProps} />);
    await user.click(within(blockNamed("Before")).getByTitle("Copy JSON"));

    await waitFor(() => expect(writeText).toHaveBeenCalledWith(JSON.stringify({ max_budget: 10 }, null, 2)));
  });
});
