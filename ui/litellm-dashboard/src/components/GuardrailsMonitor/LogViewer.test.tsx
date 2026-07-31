import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as networking from "@/components/networking";
import { LogViewer } from "./LogViewer";

vi.mock("@/components/networking", () => ({
  uiSpendLogsCall: vi.fn(),
}));

vi.mock("@/components/view_logs/LogDetailsDrawer", () => ({
  LogDetailsDrawer: ({ open, logEntry }: { open: boolean; logEntry: { request_id: string } | null }) =>
    open && logEntry ? <div data-testid="log-drawer">{logEntry.request_id}</div> : null,
}));

const mockUiSpendLogsCall = vi.mocked(networking.uiSpendLogsCall);

const originalTimezone = process.env.TZ;

function wrapper({ children }: { children: React.ReactNode }) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
}

describe("LogViewer", () => {
  beforeEach(() => {
    // UTC+2: a day window built in the wrong order collapses onto the previous day
    process.env.TZ = "Europe/Berlin";
    mockUiSpendLogsCall.mockReset();
  });

  afterEach(() => {
    process.env.TZ = originalTimezone;
  });

  it("should open the drawer for a log on the last day of the range in a UTC+ timezone", async () => {
    mockUiSpendLogsCall.mockResolvedValue({
      data: [{ request_id: "req-1" }],
      total: 1,
    });

    render(
      <LogViewer
        guardrailName="Bedrock_Test"
        logs={[
          {
            id: "req-1",
            timestamp: "2026-07-22T09:38:46.397000+00:00",
            action: "passed",
            model: "model-router",
            input_snippet: "hello",
          },
        ]}
        totalLogs={1}
        accessToken="sk-test"
        startDate="2026-07-15"
        endDate="2026-07-22"
      />,
      { wrapper },
    );

    fireEvent.click(screen.getByText("hello"));

    await waitFor(() => expect(mockUiSpendLogsCall).toHaveBeenCalled());
    const { start_date, end_date, params } = mockUiSpendLogsCall.mock.calls[0][0];
    expect(params?.request_id).toBe("req-1");
    expect(Date.parse(`${start_date}Z`)).toBeLessThanOrEqual(Date.parse("2026-07-22T09:38:46Z"));
    expect(Date.parse(`${end_date}Z`)).toBeGreaterThanOrEqual(Date.parse("2026-07-22T09:38:46Z"));

    expect(await screen.findByTestId("log-drawer")).toHaveTextContent("req-1");
  });
});
