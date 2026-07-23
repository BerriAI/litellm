import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import moment from "moment";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders, testQueryClient } from "../../../tests/test-utils";
import type { LogEntry } from "./columns";
import RequestLogsPanel from "./RequestLogsPanel";

vi.mock("../networking", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../networking")>();
  return {
    ...actual,
    uiSpendLogsCall: vi.fn(),
    keyInfoV1Call: vi.fn().mockResolvedValue({ info: {} }),
    allEndUsersCall: vi.fn().mockResolvedValue([]),
  };
});

vi.mock("@/components/key_team_helpers/filter_helpers", () => ({
  fetchAllTeams: vi.fn().mockResolvedValue([]),
}));

vi.mock("./LogDetailsDrawer", () => ({
  LogDetailsDrawer: function LogDetailsDrawerMock({ open }: { open: boolean }) {
    return <div data-testid="log-details-drawer">{open ? "open" : "closed"}</div>;
  },
}));

import { uiSpendLogsCall } from "../networking";

const logEntry = (overrides: Partial<LogEntry>): LogEntry => ({
  request_id: "req-1",
  api_key: "key-1",
  team_id: "team-1",
  model: "gpt-4o",
  model_id: "model-1",
  call_type: "acompletion",
  spend: 0.01,
  total_tokens: 10,
  prompt_tokens: 5,
  completion_tokens: 5,
  startTime: "2026-07-07T09:50:13Z",
  endTime: "2026-07-07T09:50:14Z",
  cache_hit: "false",
  messages: [],
  response: {},
  ...overrides,
});

const respondWith = (data: LogEntry[]) =>
  vi.mocked(uiSpendLogsCall).mockResolvedValue({
    data,
    total: data.length,
    page: 1,
    page_size: 50,
    total_pages: 1,
  });

const defaultProps = {
  accessToken: "test-token",
  token: "test-token",
  userRole: "Admin",
  userID: "user-1",
  isActive: true,
};

const row = (requestId: string) => document.querySelector(`[data-row-id="${requestId}"]`);
const lastCall = () => vi.mocked(uiSpendLogsCall).mock.calls.at(-1)?.[0];

describe("RequestLogsPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    sessionStorage.clear();
    testQueryClient.clear();
    respondWith([]);
  });

  describe("multi-call session collapsing", () => {
    const sessionRows = [
      logEntry({ request_id: "req-mcp", call_type: "call_mcp_tool", session_id: "sess-1", session_total_count: 3 }),
      logEntry({ request_id: "req-llm", call_type: "acompletion", session_id: "sess-1", session_total_count: 3 }),
      logEntry({ request_id: "req-llm-2", call_type: "acompletion", session_id: "sess-1", session_total_count: 3 }),
    ];

    it("collapses a multi-call session to a single representative row", async () => {
      respondWith(sessionRows);
      renderWithProviders(<RequestLogsPanel {...defaultProps} />);

      await waitFor(() => expect(row("req-mcp") ?? row("req-llm") ?? row("req-llm-2")).not.toBeNull());

      const rendered = ["req-mcp", "req-llm", "req-llm-2"].filter((id) => row(id) !== null);
      expect(rendered).toHaveLength(1);
    });

    it("prefers an LLM call over an MCP call as the session's representative", async () => {
      respondWith(sessionRows);
      renderWithProviders(<RequestLogsPanel {...defaultProps} />);

      await waitFor(() => expect(row("req-llm")).not.toBeNull());
      expect(row("req-mcp")).toBeNull();
    });

    it("shows the session's call count and composition on the representative row", async () => {
      respondWith(sessionRows);
      renderWithProviders(<RequestLogsPanel {...defaultProps} />);

      await waitFor(() => expect(row("req-llm")).not.toBeNull());
      expect(within(row("req-llm") as HTMLElement).getByText("3")).toBeInTheDocument();
    });

    it("leaves single-call rows untouched", async () => {
      respondWith([
        logEntry({ request_id: "req-solo-a", session_id: "sess-a", session_total_count: 1 }),
        logEntry({ request_id: "req-solo-b" }),
      ]);
      renderWithProviders(<RequestLogsPanel {...defaultProps} />);

      await waitFor(() => expect(row("req-solo-a")).not.toBeNull());
      expect(row("req-solo-b")).not.toBeNull();
    });
  });

  describe("client-side search", () => {
    it("narrows the visible rows without refetching", async () => {
      const user = userEvent.setup();
      respondWith([
        logEntry({ request_id: "req-alpha", model: "gpt-4o" }),
        logEntry({ request_id: "req-beta", model: "claude-opus" }),
      ]);
      renderWithProviders(<RequestLogsPanel {...defaultProps} />);

      await waitFor(() => expect(row("req-alpha")).not.toBeNull());
      const callsBefore = vi.mocked(uiSpendLogsCall).mock.calls.length;

      await user.type(screen.getByTestId("datatable-search"), "alpha");

      await waitFor(() => expect(row("req-beta")).toBeNull());
      expect(row("req-alpha")).not.toBeNull();
      expect(vi.mocked(uiSpendLogsCall).mock.calls.length).toBe(callsBefore);
    });
  });

  describe("time range", () => {
    it("requests a ~15 minute window when Last 15 Minutes is picked", async () => {
      const user = userEvent.setup();
      renderWithProviders(<RequestLogsPanel {...defaultProps} />);

      await waitFor(() => expect(uiSpendLogsCall).toHaveBeenCalled());
      await user.click(screen.getByRole("button", { name: /Last 24 Hours/i }));
      await user.click(await screen.findByRole("button", { name: "Last 15 Minutes" }));

      await waitFor(() => {
        const call = lastCall();
        if (!call) throw new Error("no call");
        const diff = moment
          .utc(call.end_date, "YYYY-MM-DD HH:mm:ss")
          .diff(moment.utc(call.start_date, "YYYY-MM-DD HH:mm:ss"), "seconds");
        expect(diff).toBeGreaterThanOrEqual(15 * 60);
        expect(diff).toBeLessThanOrEqual(16 * 60);
      });
    });

    it("restores the default 24 hour window when filters are reset", async () => {
      const user = userEvent.setup();
      renderWithProviders(<RequestLogsPanel {...defaultProps} />);

      await waitFor(() => expect(uiSpendLogsCall).toHaveBeenCalled());
      await user.click(screen.getByRole("button", { name: /Last 24 Hours/i }));
      await user.click(await screen.findByRole("button", { name: "Last 15 Minutes" }));
      await waitFor(() => expect(screen.getByRole("button", { name: /Last 15 Minutes/i })).toBeInTheDocument());

      await user.click(screen.getByRole("button", { name: "Reset Filters" }));

      expect(await screen.findByRole("button", { name: /Last 24 Hours/i })).toBeInTheDocument();

      await user.click(screen.getByTestId("datatable-refresh"));

      await waitFor(() => {
        const call = lastCall();
        if (!call) throw new Error("no call");
        const diff = moment
          .utc(call.end_date, "YYYY-MM-DD HH:mm:ss")
          .diff(moment.utc(call.start_date, "YYYY-MM-DD HH:mm:ss"), "seconds");
        expect(diff).toBeGreaterThanOrEqual(24 * 60 * 60);
      });
    });
  });

  describe("live tail", () => {
    it("shows the auto-refresh banner on the first page and hides it once stopped", async () => {
      const user = userEvent.setup();
      renderWithProviders(<RequestLogsPanel {...defaultProps} />);

      expect(await screen.findByText("Auto-refreshing every 15 seconds")).toBeInTheDocument();

      await user.click(screen.getByRole("button", { name: "Stop" }));

      expect(screen.queryByText("Auto-refreshing every 15 seconds")).toBeNull();
    });
  });
});
