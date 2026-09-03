import { QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import moment from "moment";
import { NuqsTestingAdapter, type UrlUpdateEvent } from "nuqs/adapters/testing";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { render, renderWithProviders, testQueryClient } from "../../../tests/test-utils";
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
  LogDetailsDrawer: function LogDetailsDrawerMock({
    open,
    logEntry,
    sessionId,
    onClose,
    allLogs = [],
    onSelectLog,
  }: {
    open: boolean;
    logEntry?: { request_id: string } | null;
    sessionId?: string | null;
    onClose: () => void;
    allLogs?: { request_id: string }[];
    onSelectLog?: (log: { request_id: string }) => void;
  }) {
    const nextLog = allLogs.find((log) => log.request_id !== logEntry?.request_id);
    return (
      <div data-testid="log-details-drawer" data-log-id={logEntry?.request_id ?? ""} data-session-id={sessionId ?? ""}>
        {open ? "open" : "closed"}
        <button type="button" onClick={onClose}>
          close-drawer
        </button>
        <button type="button" onClick={() => nextLog && onSelectLog?.(nextLog)}>
          select-next-log
        </button>
      </div>
    );
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

const onUrlUpdate = vi.fn<(event: UrlUpdateEvent) => void>();
const renderPanel = (searchParams?: string) =>
  renderWithProviders(<RequestLogsPanel {...defaultProps} />, { searchParams, onUrlUpdate });

const renderPanelWithHistory = () => {
  const stack = [""];
  const handleUrlUpdate = (event: UrlUpdateEvent) => {
    onUrlUpdate(event);
    if (event.options.history === "push") {
      stack.push(event.queryString);
    } else {
      stack[stack.length - 1] = event.queryString;
    }
  };
  const tree = (searchParams: string) => (
    <NuqsTestingAdapter searchParams={searchParams} onUrlUpdate={handleUrlUpdate} hasMemory>
      <QueryClientProvider client={testQueryClient}>
        <RequestLogsPanel {...defaultProps} />
      </QueryClientProvider>
    </NuqsTestingAdapter>
  );
  const view = render(tree(""));
  return {
    goBack: () => {
      const current = stack[stack.length - 1] ?? "";
      stack.pop();
      const target = stack[stack.length - 1] ?? "";
      view.rerender(tree(current));
      view.rerender(tree(target));
    },
  };
};
const urlParams = () => onUrlUpdate.mock.calls.at(-1)?.[0].searchParams ?? new URLSearchParams();
const historyModes = () => onUrlUpdate.mock.calls.map(([event]) => event.options.history);

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
      renderPanel();

      await waitFor(() => expect(row("req-mcp") ?? row("req-llm") ?? row("req-llm-2")).not.toBeNull());

      const rendered = ["req-mcp", "req-llm", "req-llm-2"].filter((id) => row(id) !== null);
      expect(rendered).toHaveLength(1);
    });

    it("prefers an LLM call over an MCP call as the session's representative", async () => {
      respondWith(sessionRows);
      renderPanel();

      await waitFor(() => expect(row("req-llm")).not.toBeNull());
      expect(row("req-mcp")).toBeNull();
    });

    it("shows the session's call count and composition on the representative row", async () => {
      respondWith(sessionRows);
      renderPanel();

      await waitFor(() => expect(row("req-llm")).not.toBeNull());
      expect(within(row("req-llm") as HTMLElement).getByText("3")).toBeInTheDocument();
    });

    it("leaves single-call rows untouched", async () => {
      respondWith([
        logEntry({ request_id: "req-solo-a", session_id: "sess-a", session_total_count: 1 }),
        logEntry({ request_id: "req-solo-b" }),
      ]);
      renderPanel();

      await waitFor(() => expect(row("req-solo-a")).not.toBeNull());
      expect(row("req-solo-b")).not.toBeNull();
    });
  });

  describe("search by request id (LIT-3981)", () => {
    it("sends the typed request id to the server on the first page instead of filtering the loaded rows", async () => {
      const user = userEvent.setup();
      renderPanel();

      await waitFor(() => expect(uiSpendLogsCall).toHaveBeenCalled());

      fireEvent.change(screen.getByTestId("datatable-search"), { target: { value: "req-on-another-page" } });

      await waitFor(() => {
        const call = lastCall();
        if (!call) throw new Error("uiSpendLogsCall was not called");
        expect(call.params?.request_id).toBe("req-on-another-page");
        expect(call.page).toBe(1);
      });
    });
  });

  describe("time range", () => {
    it("requests a ~15 minute window when Last 15 Minutes is picked", async () => {
      const user = userEvent.setup();
      renderPanel();

      await waitFor(() => expect(uiSpendLogsCall).toHaveBeenCalled());
      await user.click(screen.getByRole("button", { name: /Last 24 Hours/i }));
      await user.click(await screen.findByRole("button", { name: "Last 15 Minutes" }));

      const windowSeconds = () => {
        const call = lastCall();
        if (!call) throw new Error("no call");
        return moment
          .utc(call.end_date, "YYYY-MM-DD HH:mm:ss")
          .diff(moment.utc(call.start_date, "YYYY-MM-DD HH:mm:ss"), "seconds");
      };

      await waitFor(() => expect(windowSeconds()).toBeGreaterThanOrEqual(15 * 60));
      expect(windowSeconds()).toBeLessThanOrEqual(16 * 60);
    });

    it("restores the default 24 hour window when filters are reset", async () => {
      const user = userEvent.setup();
      renderPanel();

      await waitFor(() => expect(uiSpendLogsCall).toHaveBeenCalled());
      await user.click(screen.getByRole("button", { name: /Last 24 Hours/i }));
      await user.click(await screen.findByRole("button", { name: "Last 15 Minutes" }));
      expect(await screen.findByRole("button", { name: /Last 15 Minutes/i })).toBeInTheDocument();

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

  describe("shareable log links (?log_id=)", () => {
    const drawer = () => screen.getByTestId("log-details-drawer");

    it("clicking a row writes ?log_id=<request_id> to the URL and opens the drawer", async () => {
      const user = userEvent.setup();
      respondWith([logEntry({ request_id: "req-1" })]);
      renderPanel();

      await waitFor(() => expect(row("req-1")).not.toBeNull());
      await user.click(row("req-1") as HTMLElement);

      await waitFor(() => expect(urlParams().get("log_id")).toBe("req-1"));
      expect(historyModes()).toEqual(["push"]);
      await waitFor(() => {
        expect(drawer()).toHaveTextContent("open");
      });
      expect(drawer()).toHaveAttribute("data-log-id", "req-1");
    });

    it("opens the drawer on load when ?log_id= matches a log in the loaded page", async () => {
      respondWith([logEntry({ request_id: "req-1" }), logEntry({ request_id: "req-2" })]);
      renderPanel("?log_id=req-2");

      await waitFor(() => {
        expect(drawer()).toHaveTextContent("open");
      });
      expect(drawer()).toHaveAttribute("data-log-id", "req-2");
    });

    it("fetches the log by request_id and opens the drawer when it is not in the loaded page", async () => {
      vi.mocked(uiSpendLogsCall).mockImplementation(async ({ params }) =>
        params?.request_id === "req-old"
          ? { data: [logEntry({ request_id: "req-old" })], total: 1, page: 1, page_size: 1, total_pages: 1 }
          : { data: [], total: 0, page: 1, page_size: 50, total_pages: 0 },
      );
      renderPanel("?log_id=req-old");

      await waitFor(() => {
        expect(drawer()).toHaveTextContent("open");
      });
      expect(drawer()).toHaveAttribute("data-log-id", "req-old");

      const byIdCall = vi
        .mocked(uiSpendLogsCall)
        .mock.calls.find(([options]) => options.params?.request_id === "req-old")?.[0];
      if (!byIdCall) throw new Error("expected a by-id uiSpendLogsCall");
      expect(byIdCall.page).toBe(1);
      expect(byIdCall.page_size).toBe(1);
    });

    it("closing the drawer removes ?log_id= from the URL and closes the drawer", async () => {
      const user = userEvent.setup();
      respondWith([logEntry({ request_id: "req-1" })]);
      renderPanel();

      await waitFor(() => expect(row("req-1")).not.toBeNull());
      await user.click(row("req-1") as HTMLElement);
      await waitFor(() => expect(drawer()).toHaveTextContent("open"));

      await user.click(screen.getByRole("button", { name: "close-drawer" }));

      await waitFor(() => expect(urlParams().get("log_id")).toBeNull());
      await waitFor(() => expect(drawer()).toHaveTextContent("closed"));
    });

    it("switching logs inside the drawer replaces the URL, so back closes the drawer in one step", async () => {
      const user = userEvent.setup();
      respondWith([logEntry({ request_id: "req-1" }), logEntry({ request_id: "req-2" })]);
      const { goBack } = renderPanelWithHistory();

      await waitFor(() => expect(row("req-1")).not.toBeNull());
      await user.click(row("req-1") as HTMLElement);
      await waitFor(() => expect(drawer()).toHaveAttribute("data-log-id", "req-1"));

      await user.click(screen.getByRole("button", { name: "select-next-log" }));
      await waitFor(() => expect(drawer()).toHaveAttribute("data-log-id", "req-2"));
      expect(urlParams().get("log_id")).toBe("req-2");
      expect(historyModes()).toEqual(["push", "replace"]);

      goBack();
      await waitFor(() => expect(drawer()).toHaveTextContent("closed"));
      expect(drawer()).toHaveAttribute("data-log-id", "");
    });

    it("clicking a session id writes ?session_id= and ?log_id= and opens the session drawer", async () => {
      const user = userEvent.setup();
      respondWith([logEntry({ request_id: "req-solo", session_id: "sess-solo", session_total_count: 1 })]);
      renderPanel();

      await waitFor(() => expect(row("req-solo")).not.toBeNull());
      await user.click(within(row("req-solo") as HTMLElement).getByText("sess-solo"));

      await waitFor(() => expect(urlParams().get("session_id")).toBe("sess-solo"));
      expect(urlParams().get("log_id")).toBe("req-solo");
      expect(historyModes()).toEqual(["push"]);
      await waitFor(() => {
        expect(drawer()).toHaveTextContent("open");
      });
      expect(drawer()).toHaveAttribute("data-session-id", "sess-solo");
    });

    it("clicking a log row clears a lingering ?session_id= so the drawer shows the clicked log", async () => {
      const user = userEvent.setup();
      respondWith([
        logEntry({ request_id: "req-a", session_id: "sess-a", session_total_count: 1 }),
        logEntry({ request_id: "req-b" }),
      ]);
      renderPanel();

      await waitFor(() => expect(row("req-a")).not.toBeNull());
      await user.click(within(row("req-a") as HTMLElement).getByText("sess-a"));
      await waitFor(() => expect(urlParams().get("session_id")).toBe("sess-a"));

      await user.click(row("req-b") as HTMLElement);

      await waitFor(() => expect(urlParams().get("log_id")).toBe("req-b"));
      expect(urlParams().get("session_id")).toBeNull();
      await waitFor(() => {
        expect(drawer()).toHaveAttribute("data-log-id", "req-b");
      });
      expect(drawer()).toHaveAttribute("data-session-id", "");
    });

    it("closing a drawer opened via a session id clears both params", async () => {
      const user = userEvent.setup();
      respondWith([logEntry({ request_id: "req-solo", session_id: "sess-solo", session_total_count: 1 })]);
      renderPanel();

      await waitFor(() => expect(row("req-solo")).not.toBeNull());
      await user.click(within(row("req-solo") as HTMLElement).getByText("sess-solo"));
      await waitFor(() => expect(drawer()).toHaveTextContent("open"));

      await user.click(screen.getByRole("button", { name: "close-drawer" }));

      await waitFor(() => expect(drawer()).toHaveTextContent("closed"));
      expect(urlParams().get("session_id")).toBeNull();
      expect(urlParams().get("log_id")).toBeNull();
    });

    it("opens a deep-linked multi-call session log in session mode", async () => {
      respondWith([
        logEntry({ request_id: "req-llm", call_type: "acompletion", session_id: "sess-1", session_total_count: 3 }),
      ]);
      renderPanel("?log_id=req-llm");

      await waitFor(() => {
        expect(drawer()).toHaveTextContent("open");
      });
      expect(drawer()).toHaveAttribute("data-session-id", "sess-1");
      expect(drawer()).toHaveAttribute("data-log-id", "req-llm");
    });

    it("clicking a multi-call session's row writes ?session_id= alongside ?log_id=", async () => {
      const user = userEvent.setup();
      respondWith([
        logEntry({ request_id: "req-llm", call_type: "acompletion", session_id: "sess-1", session_total_count: 3 }),
        logEntry({ request_id: "req-llm-2", call_type: "acompletion", session_id: "sess-1", session_total_count: 3 }),
      ]);
      renderPanel();

      await waitFor(() => expect(row("req-llm")).not.toBeNull());
      await user.click(row("req-llm") as HTMLElement);

      await waitFor(() => expect(urlParams().get("session_id")).toBe("sess-1"));
      expect(urlParams().get("log_id")).toBe("req-llm");
      await waitFor(() => expect(drawer()).toHaveAttribute("data-session-id", "sess-1"));
    });

    it("selecting another log while a session view is open keeps the session open", async () => {
      const user = userEvent.setup();
      respondWith([
        logEntry({ request_id: "req-llm", call_type: "acompletion", session_id: "sess-1", session_total_count: 3 }),
        logEntry({ request_id: "req-unenriched" }),
      ]);
      renderPanel("?log_id=req-llm");

      await waitFor(() => expect(drawer()).toHaveAttribute("data-session-id", "sess-1"));

      await user.click(screen.getByRole("button", { name: "select-next-log" }));

      await waitFor(() => expect(drawer()).toHaveAttribute("data-log-id", "req-unenriched"));
      expect(urlParams().get("session_id")).toBe("sess-1");
      expect(drawer()).toHaveAttribute("data-session-id", "sess-1");
    });
  });

  describe("hide health checks", () => {
    const toggle = () => screen.getByRole("switch", { name: "Hide Health Checks" });

    it("defaults to showing health checks and refetches without them from page 1 when toggled on", async () => {
      const user = userEvent.setup();
      renderPanel();

      await waitFor(() => expect(uiSpendLogsCall).toHaveBeenCalled());
      expect(lastCall()?.params?.exclude_internal_health_checks).toBe(false);
      expect(toggle()).not.toBeChecked();

      await user.click(toggle());

      await waitFor(() => expect(lastCall()?.params?.exclude_internal_health_checks).toBe(true));
      expect(lastCall()?.page).toBe(1);
      expect(toggle()).toBeChecked();
      expect(sessionStorage.getItem("excludeInternalHealthChecks")).toBe("true");
    });

    it("restores the persisted toggle from sessionStorage", async () => {
      sessionStorage.setItem("excludeInternalHealthChecks", "true");
      renderPanel();

      await waitFor(() => expect(uiSpendLogsCall).toHaveBeenCalled());
      expect(lastCall()?.params?.exclude_internal_health_checks).toBe(true);
      expect(toggle()).toBeChecked();
    });

    it("falls back to showing health checks when the persisted value is malformed", async () => {
      sessionStorage.setItem("excludeInternalHealthChecks", "{not json");
      renderPanel();

      await waitFor(() => expect(uiSpendLogsCall).toHaveBeenCalled());
      expect(lastCall()?.params?.exclude_internal_health_checks).toBe(false);
      expect(toggle()).not.toBeChecked();
    });
  });

  describe("live tail", () => {
    it("shows the auto-refresh banner on the first page and hides it once stopped", async () => {
      const user = userEvent.setup();
      renderPanel();

      expect(await screen.findByText("Auto-refreshing every 15 seconds")).toBeInTheDocument();

      await user.click(screen.getByRole("button", { name: "Stop" }));

      expect(screen.queryByText("Auto-refreshing every 15 seconds")).not.toBeInTheDocument();
    });
  });
});
