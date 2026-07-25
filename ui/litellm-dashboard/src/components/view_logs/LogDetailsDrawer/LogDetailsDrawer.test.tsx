import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { LogDetailsDrawer } from "./LogDetailsDrawer";
import { sessionSpendLogsCall } from "../../networking";
import { LogEntry } from "../columns";
import { AutoRouterModelGroupsProvider } from "@/components/shared/table_cells";

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

vi.mock("@/app/(dashboard)/hooks/models/useModels", () => ({
  useAutoRouterModelGroups: vi.fn(() => new Set(["smart-router"])),
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
    endTime: "2026-07-08T10:00:06.000Z",
  }),
  makeLog({
    request_id: "llm-late",
    model: "llm-late",
    startTime: "2026-07-08T10:00:02.000Z",
    endTime: "2026-07-08T10:00:05.000Z",
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
  const drawer = (open: boolean) => (
    <QueryClientProvider client={queryClient}>
      <LogDetailsDrawer open={open} onClose={() => {}} logEntry={null} sessionId="session-1" accessToken="token" />
    </QueryClientProvider>
  );
  const { rerender } = render(drawer(true));
  return { rerender, drawer };
};

const sidebarEventNames = () =>
  screen.queryAllByText(/^(llm-early|llm-late|tool-early|tool-late)$/).map((el) => el.textContent);

describe("LogDetailsDrawer session sidebar sorting", () => {
  it("defaults to duration order, longest call first across LLM and MCP calls", async () => {
    renderSessionDrawer();
    await waitFor(() => expect(sidebarEventNames()).toHaveLength(4));
    expect(sidebarEventNames()).toEqual(["tool-early", "llm-late", "llm-early", "tool-late"]);
  });

  it("switches to chronological order across LLM and MCP calls when Start time is selected", async () => {
    renderSessionDrawer();
    await waitFor(() => expect(sidebarEventNames()).toHaveLength(4));

    fireEvent.click(screen.getByText("Start time"));

    await waitFor(() => expect(sidebarEventNames()).toEqual(["llm-early", "tool-early", "llm-late", "tool-late"]));

    fireEvent.click(screen.getByText("Duration"));

    await waitFor(() => expect(sidebarEventNames()).toEqual(["tool-early", "llm-late", "llm-early", "tool-late"]));
  });

  it("resets the sort mode back to duration when the drawer is closed and reopened", async () => {
    const { rerender, drawer } = renderSessionDrawer();
    await waitFor(() => expect(sidebarEventNames()).toHaveLength(4));

    fireEvent.click(screen.getByText("Start time"));
    await waitFor(() => expect(sidebarEventNames()).toEqual(["llm-early", "tool-early", "llm-late", "tool-late"]));

    rerender(drawer(false));
    rerender(drawer(true));

    await waitFor(() => expect(sidebarEventNames()).toEqual(["tool-early", "llm-late", "llm-early", "tool-late"]));
  });
});

describe("LogDetailsDrawer session sidebar auto-router icon", () => {
  const routedSessionLogs = [
    makeLog({ request_id: "routed", model: "claude-opus-4-8", model_group: "smart-router" }),
    makeLog({ request_id: "direct", model: "claude-haiku-4-5", model_group: "claude-haiku" }),
  ];

  const renderRoutedSession = () => {
    vi.mocked(sessionSpendLogsCall).mockResolvedValue({ data: routedSessionLogs, total: 2, total_pages: 1 });
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    return render(
      <QueryClientProvider client={queryClient}>
        <AutoRouterModelGroupsProvider>
          <LogDetailsDrawer open onClose={() => {}} logEntry={null} sessionId="session-1" accessToken="token" />
        </AutoRouterModelGroupsProvider>
      </QueryClientProvider>,
    );
  };

  const rowFor = (label: string): HTMLElement => {
    const row = Array.from(document.body.querySelectorAll("button")).find((button) =>
      button.textContent?.includes(label),
    );
    if (!row) throw new Error(`no sidebar row for ${label}`);
    return row;
  };

  it("marks the auto-routed entry with the router icon and leaves a direct call on the default icon", async () => {
    renderRoutedSession();

    await waitFor(() => expect(screen.queryByText("claude-opus-4-8")).not.toBeNull());

    const routedRow = rowFor("claude-opus-4-8");
    const directRow = rowFor("claude-haiku-4-5");

    expect(routedRow.querySelector(".lucide-waypoints")).not.toBeNull();
    expect(routedRow.querySelector(".lucide-sparkles")).toBeNull();
    expect(directRow.querySelector(".lucide-sparkles")).not.toBeNull();
    expect(directRow.querySelector(".lucide-waypoints")).toBeNull();
  });
});
