import { act, render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it, vi } from "vitest";
import { LogViewer } from "./LogViewer";
import type { LogEntry } from "./mockData";

vi.mock("@/components/view_logs/LogDetailsDrawer", () => ({
  LogDetailsDrawer: ({
    open,
    onClose,
  }: {
    open: boolean;
    onClose: () => void;
  }) =>
    open ? (
      <div data-testid="log-details-drawer" role="dialog">
        Log details drawer
        <button type="button" onClick={onClose}>
          Close log drawer
        </button>
      </div>
    ) : null,
}));

vi.mock("./AgentTrace", () => ({
  AgentTraceDrawer: ({
    open,
    onClose,
    session,
  }: {
    open: boolean;
    onClose: () => void;
    session: { rootAgentName: string; shortId: string } | null;
  }) =>
    open && session ? (
      <div data-testid="agent-trace-drawer" role="dialog">
        Agent trace: {session.rootAgentName}
        <p>Session: {session.shortId}</p>
        <button type="button" onClick={onClose}>
          Close trace drawer
        </button>
      </div>
    ) : null,
}));

const mockLogs: LogEntry[] = [
  {
    id: "log-1",
    timestamp: "2026-03-02 21:00:00",
    action: "passed",
    model: "gpt-4o",
    input_snippet: "First guardrail log request",
  },
  {
    id: "log-2",
    timestamp: "2026-03-02 21:01:00",
    action: "blocked",
    model: "claude-3",
    input_snippet: "Second guardrail log request",
  },
  {
    id: "log-3",
    timestamp: "2026-03-02 21:02:00",
    action: "flagged",
    input_snippet: "Third guardrail log request",
  },
];

function wrapper({ children }: { children: React.ReactNode }) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });
  return (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  );
}

describe("LogViewer", () => {
  it("should render agent-trace demo rows first then guardrail log entries", () => {
    render(
      <LogViewer
        guardrailName="Test Guardrail"
        logs={mockLogs}
        logsLoading={false}
      />,
      { wrapper }
    );

    // First 2 rows are the fixed agent-trace demo rows
    expect(screen.getByText("Currency Research Agent")).toBeDefined();
    expect(screen.getByText("Travel Booking Agent")).toBeDefined();
    // Trace badge appears for demo rows
    const traceBadges = screen.getAllByText("Trace");
    expect(traceBadges.length).toBeGreaterThanOrEqual(2);

    // Following rows are guardrail log entries (input snippets are unique to log rows)
    expect(screen.getByText("First guardrail log request")).toBeDefined();
    expect(screen.getByText("Second guardrail log request")).toBeDefined();
    expect(screen.getByText("Third guardrail log request")).toBeDefined();
  });

  it("should open AgentTraceDrawer when first demo trace row is clicked", async () => {
    render(
      <LogViewer
        guardrailName="Test Guardrail"
        logs={mockLogs}
        logsLoading={false}
      />,
      { wrapper }
    );

    const currencyRow = screen.getByText("Currency Research Agent");
    expect(screen.queryByTestId("agent-trace-drawer")).toBeNull();

    await act(async () => {
      currencyRow.closest("button")?.click();
    });

    expect(screen.getByTestId("agent-trace-drawer")).toBeDefined();
    expect(screen.getByText(/Agent trace: Currency Research Agent/)).toBeDefined();
    expect(screen.getByText(/Session: 0c4b4759-83aa/)).toBeDefined();
  });
});
