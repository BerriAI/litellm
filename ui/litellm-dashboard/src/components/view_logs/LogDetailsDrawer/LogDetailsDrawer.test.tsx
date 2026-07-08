import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { LogDetailsDrawer } from "./LogDetailsDrawer";
import { sessionSpendLogsCall } from "../../networking";
import { LogEntry } from "../columns";

vi.mock("../../networking", () => ({
  sessionSpendLogsCall: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/logDetails/useLogDetails", () => ({
  useLogDetails: () => ({ data: null, isLoading: false }),
}));

vi.mock("./LogDetailContent", () => ({
  LogDetailContent: () => null,
  GuardrailJumpLink: () => null,
}));

vi.mock("./DrawerHeader", () => ({
  DrawerHeader: () => null,
}));

const makeLog = (overrides: Partial<LogEntry>): LogEntry => ({
  request_id: "req",
  api_key: "",
  team_id: "",
  model: "",
  model_id: "",
  call_type: "acompletion",
  spend: 0,
  total_tokens: 0,
  prompt_tokens: 0,
  completion_tokens: 0,
  startTime: "2026-07-08T10:00:00.000Z",
  endTime: "2026-07-08T10:00:01.000Z",
  cache_hit: "false",
  messages: [],
  response: {},
  ...overrides,
});

const sessionLogs = [
  makeLog({
    request_id: "llm-early",
    model: "llm-early",
    startTime: "2026-07-08T10:00:00.000Z",
    endTime: "2026-07-08T10:00:02.000Z",
  }),
  makeLog({
    request_id: "mcp-early",
    model: "tool-early",
    call_type: "call_mcp_tool",
    startTime: "2026-07-08T10:00:01.000Z",
    endTime: "2026-07-08T10:00:01.500Z",
  }),
  makeLog({
    request_id: "llm-late",
    model: "llm-late",
    startTime: "2026-07-08T10:00:02.000Z",
    endTime: "2026-07-08T10:00:04.000Z",
  }),
  makeLog({
    request_id: "mcp-late",
    model: "tool-late",
    call_type: "call_mcp_tool",
    startTime: "2026-07-08T10:00:03.000Z",
    endTime: "2026-07-08T10:00:03.500Z",
  }),
];

const renderSessionDrawer = () => {
  vi.mocked(sessionSpendLogsCall).mockResolvedValue({ data: sessionLogs, total: 4, total_pages: 1 });
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={queryClient}>
      <LogDetailsDrawer open onClose={() => {}} logEntry={null} sessionId="session-1" accessToken="token" />
    </QueryClientProvider>,
  );
};

const sidebarEventNames = () =>
  screen.queryAllByText(/^(llm-early|llm-late|tool-early|tool-late)$/).map((el) => el.textContent);

describe("LogDetailsDrawer session sidebar sorting", () => {
  it("defaults to grouped order: LLM calls newest first, MCP calls grouped last", async () => {
    renderSessionDrawer();
    await waitFor(() => expect(sidebarEventNames()).toHaveLength(4));
    expect(sidebarEventNames()).toEqual(["llm-late", "llm-early", "tool-late", "tool-early"]);
  });

  it("switches to chronological order across LLM and MCP calls when Start time is selected", async () => {
    renderSessionDrawer();
    await waitFor(() => expect(sidebarEventNames()).toHaveLength(4));

    fireEvent.click(screen.getByText("Start time"));

    await waitFor(() => expect(sidebarEventNames()).toEqual(["llm-early", "tool-early", "llm-late", "tool-late"]));

    fireEvent.click(screen.getByText("Grouped"));

    await waitFor(() => expect(sidebarEventNames()).toEqual(["llm-late", "llm-early", "tool-late", "tool-early"]));
  });
});
