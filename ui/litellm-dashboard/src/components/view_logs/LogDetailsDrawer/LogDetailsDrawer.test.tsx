import { screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { LogDetailsDrawer } from "./LogDetailsDrawer";
import { renderWithProviders } from "../../../../tests/test-utils";
import { sessionSpendLogsCall } from "../../networking";
import type { LogEntry } from "../columns";

vi.mock("../../networking", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../networking")>();
  return {
    ...actual,
    sessionSpendLogsCall: vi.fn(),
  };
});

vi.mock("@/app/(dashboard)/hooks/logDetails/useLogDetails", () => ({
  useLogDetails: () => ({
    data: { messages: [], response: {} },
    isLoading: false,
  }),
}));

const makeLog = (overrides: Partial<LogEntry>): LogEntry =>
  ({
    request_id: "req-1",
    api_key: "key-hash",
    team_id: "team-1",
    model: "gpt-4o",
    model_id: "model-1",
    call_type: "acompletion",
    spend: 0.01,
    total_tokens: 10,
    prompt_tokens: 5,
    completion_tokens: 5,
    startTime: "2026-07-13 20:00:00",
    endTime: "2026-07-13 20:00:01",
    user: "user-1",
    cache_hit: "None",
    messages: [],
    response: {},
    session_id: "session-a",
    status: "success",
    ...overrides,
  }) as LogEntry;

describe("LogDetailsDrawer session selection", () => {
  const olderLog = makeLog({
    request_id: "req-old",
    model: "claude-sonnet-4-6",
    startTime: "2026-07-13 20:00:00",
  });
  const latestLog = makeLog({
    request_id: "req-latest",
    model: "claude-sonnet-5",
    startTime: "2026-07-13 20:01:00",
  });
  const mcpLog = makeLog({
    request_id: "req-mcp",
    model: "mcp:jira/search",
    call_type: "call_mcp_tool",
    startTime: "2026-07-13 20:00:30",
  });

  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(sessionSpendLogsCall).mockResolvedValue({
      data: [olderLog, mcpLog, latestLog],
      total: 3,
      page: 1,
      page_size: 100,
      total_pages: 1,
    });
  });

  it("defaults the selected session request to the most recent log, not the clicked logEntry", async () => {
    renderWithProviders(
      <LogDetailsDrawer
        open
        onClose={vi.fn()}
        logEntry={olderLog}
        sessionId="session-a"
        accessToken="test-token"
      />,
    );

    const drawer = await screen.findByRole("dialog");
    await waitFor(() => {
      expect(within(drawer).getByText("req-latest")).toBeInTheDocument();
    });

    const selectedSessionItem = within(drawer)
      .getAllByRole("button")
      .find((button) => button.className.includes("bg-blue-50"));
    expect(selectedSessionItem).toBeTruthy();
    expect(selectedSessionItem).toHaveTextContent("claude-sonnet-5");
    expect(selectedSessionItem).not.toHaveTextContent("claude-sonnet-4-6");
  });
});
