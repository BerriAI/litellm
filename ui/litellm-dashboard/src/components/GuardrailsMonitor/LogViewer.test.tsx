import userEvent from "@testing-library/user-event";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders, screen, testQueryClient, waitFor, within } from "../../../tests/test-utils";
import type { LogEntry as SpendLogEntry } from "@/components/view_logs/columns";
import { LogViewer } from "./LogViewer";
import type { LogViewerState } from "./useLogViewerState";

vi.mock("@/components/networking", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/components/networking")>();
  return { ...actual, uiSpendLogsCall: vi.fn() };
});

vi.mock("@/components/view_logs/LogDetailsDrawer", () => ({
  LogDetailsDrawer: function LogDetailsDrawerMock({
    open,
    onClose,
    logEntry,
  }: {
    open: boolean;
    onClose: () => void;
    logEntry?: { request_id: string } | null;
  }) {
    return (
      <div data-testid="log-details-drawer" data-log-id={logEntry?.request_id ?? ""}>
        {open ? "open" : "closed"}
        <button onClick={onClose}>close drawer</button>
      </div>
    );
  },
}));

import { uiSpendLogsCall } from "@/components/networking";

const spendLog = (overrides: Partial<SpendLogEntry>): SpendLogEntry => ({
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
  startTime: "2026-09-02T09:50:13Z",
  endTime: "2026-09-02T09:50:14Z",
  cache_hit: "false",
  messages: [],
  response: {},
  ...overrides,
});

const guardrailLog = {
  id: "provider-victim",
  timestamp: "2026-09-02 09:50:13",
  action: "passed" as const,
  input_snippet: "victim prompt",
};

describe("GuardrailsMonitor LogViewer drawer", () => {
  beforeEach(() => {
    vi.mocked(uiSpendLogsCall).mockReset();
    testQueryClient.clear();
  });

  it("opens the row whose request_id is the clicked log id even when a newer row carries that id as its call id", async () => {
    vi.mocked(uiSpendLogsCall).mockResolvedValue({
      data: [
        spendLog({ request_id: "provider-attacker", litellm_call_id: "provider-victim" }),
        spendLog({ request_id: "provider-victim", litellm_call_id: "call-victim" }),
      ],
      total: 2,
    });

    renderWithProviders(<LogViewer logs={[guardrailLog]} accessToken="sk-test" />);
    await userEvent.click(screen.getByText("victim prompt"));

    await waitFor(() => {
      expect(screen.getByTestId("log-details-drawer")).toHaveAttribute("data-log-id", "provider-victim");
    });
    expect(vi.mocked(uiSpendLogsCall)).toHaveBeenCalledWith(
      expect.objectContaining({ params: { request_id: "provider-victim" } }),
    );
  });

  it("falls back to the first returned row when none carries the clicked id as its request_id", async () => {
    vi.mocked(uiSpendLogsCall).mockResolvedValue({
      data: [spendLog({ request_id: "provider-other", litellm_call_id: "provider-victim" })],
      total: 1,
    });

    renderWithProviders(<LogViewer logs={[guardrailLog]} accessToken="sk-test" />);
    await userEvent.click(screen.getByText("victim prompt"));

    await waitFor(() => {
      expect(screen.getByTestId("log-details-drawer")).toHaveAttribute("data-log-id", "provider-other");
    });
  });
});

describe("GuardrailsMonitor LogViewer not_run rows", () => {
  it("renders a not_run log as a neutral Not run badge instead of a pass or failure", () => {
    renderWithProviders(
      <LogViewer logs={[{ ...guardrailLog, action: "not_run", input_snippet: "system prompt only" }]} />,
    );

    const row = screen.getByRole("button", { name: /system prompt only/ });
    expect(within(row).getByText("Not run")).toHaveClass("text-muted-foreground");
    expect(within(row).queryByText("Passed")).not.toBeInTheDocument();
    expect(within(row).queryByText("Blocked")).not.toBeInTheDocument();
  });
});

describe("GuardrailsMonitor LogViewer view state", () => {
  const logs = [
    { id: "log-passed", timestamp: "2026-09-02 09:50:13", action: "passed" as const, input_snippet: "safe prompt" },
    { id: "log-blocked", timestamp: "2026-09-02 09:51:13", action: "blocked" as const, input_snippet: "bad prompt" },
  ];
  const makeViewState = (overrides: Partial<LogViewerState> = {}): LogViewerState => ({
    filter: "all",
    setFilter: vi.fn(),
    sampleSize: 10,
    setSampleSize: vi.fn(),
    requestId: null,
    setRequestId: vi.fn(),
    ...overrides,
  });

  beforeEach(() => {
    vi.mocked(uiSpendLogsCall).mockReset();
    vi.mocked(uiSpendLogsCall).mockResolvedValue({ data: [], total: 0 });
    testQueryClient.clear();
  });

  it("shows the filter, sample size and open request that the caller passes in", () => {
    const viewState = makeViewState({ filter: "blocked", sampleSize: 50, requestId: "log-blocked" });
    renderWithProviders(<LogViewer logs={logs} accessToken="sk-test" viewState={viewState} />);

    expect(screen.queryByText("safe prompt")).not.toBeInTheDocument();
    expect(screen.getByText("bad prompt")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "50" })).toHaveClass("bg-primary");
    expect(screen.getByRole("button", { name: "10" })).not.toHaveClass("bg-primary");
    expect(screen.getByRole("button", { name: "Blocked" })).toHaveClass("bg-primary");
    expect(screen.getByTestId("log-details-drawer")).toHaveTextContent("open");
    expect(vi.mocked(uiSpendLogsCall)).toHaveBeenCalledWith(
      expect.objectContaining({ params: { request_id: "log-blocked" } }),
    );
  });

  it("reports filter, sample size, row and drawer changes to the caller instead of keeping them", async () => {
    const user = userEvent.setup();
    const viewState = makeViewState();
    renderWithProviders(<LogViewer logs={logs} accessToken="sk-test" viewState={viewState} />);

    await user.click(screen.getByRole("button", { name: "Blocked" }));
    expect(viewState.setFilter).toHaveBeenCalledWith("blocked");
    expect(screen.getByText("safe prompt")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "100" }));
    expect(viewState.setSampleSize).toHaveBeenCalledWith(100);

    await user.click(screen.getByText("bad prompt"));
    expect(viewState.setRequestId).toHaveBeenLastCalledWith("log-blocked");
    expect(screen.getByTestId("log-details-drawer")).toHaveTextContent("closed");

    await user.click(screen.getByRole("button", { name: "close drawer" }));
    expect(viewState.setRequestId).toHaveBeenLastCalledWith(null);
  });

  it("keeps its own filter and drawer state when the caller passes none", async () => {
    const user = userEvent.setup();
    renderWithProviders(<LogViewer logs={logs} accessToken="sk-test" />);

    await user.click(screen.getByRole("button", { name: "Blocked" }));
    expect(screen.queryByText("safe prompt")).not.toBeInTheDocument();
    expect(screen.getByText("bad prompt")).toBeInTheDocument();

    await user.click(screen.getByText("bad prompt"));
    expect(screen.getByTestId("log-details-drawer")).toHaveTextContent("open");

    await user.click(screen.getByRole("button", { name: "close drawer" }));
    expect(screen.getByTestId("log-details-drawer")).toHaveTextContent("closed");
  });

  it("starts on the filter given by filterAction when the caller passes no view state", () => {
    renderWithProviders(<LogViewer logs={logs} filterAction="passed" />);

    expect(screen.getByText("safe prompt")).toBeInTheDocument();
    expect(screen.queryByText("bad prompt")).not.toBeInTheDocument();
  });
});
