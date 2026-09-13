import userEvent from "@testing-library/user-event";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders, screen, testQueryClient, waitFor } from "../../../tests/test-utils";
import type { LogEntry as SpendLogEntry } from "@/components/view_logs/columns";
import { LogViewer } from "./LogViewer";

vi.mock("@/components/networking", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/components/networking")>();
  return { ...actual, uiSpendLogsCall: vi.fn() };
});

vi.mock("@/components/view_logs/LogDetailsDrawer", () => ({
  LogDetailsDrawer: function LogDetailsDrawerMock({
    open,
    logEntry,
  }: {
    open: boolean;
    logEntry?: { request_id: string } | null;
  }) {
    return (
      <div data-testid="log-details-drawer" data-log-id={logEntry?.request_id ?? ""}>
        {open ? "open" : "closed"}
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
