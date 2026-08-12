import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import moment from "moment";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import { LogViewer } from "./LogViewer";
import { uiSpendLogsCall } from "@/components/networking";
import type { LogEntry } from "./mockData";

vi.mock("@/components/networking", () => ({
  uiSpendLogsCall: vi.fn(),
}));

vi.mock("@/components/view_logs/LogDetailsDrawer", () => ({
  LogDetailsDrawer: () => null,
}));

const originalTimezone = process.env.TZ;

const logs: LogEntry[] = [
  {
    id: "chatcmpl-1",
    timestamp: "2026-07-22T09:38:46.397000+00:00",
    action: "passed",
    model: "claude-haiku-4-5",
    input_snippet: "hello",
  },
];

const renderViewer = (startDate: string, endDate: string) => {
  vi.mocked(uiSpendLogsCall).mockResolvedValue({ data: [], total: 0 });
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <LogViewer logs={logs} accessToken="token" startDate={startDate} endDate={endDate} />
    </QueryClientProvider>,
  );
};

describe("LogViewer log lookup window", () => {
  beforeAll(() => {
    // Reproduces a UTC+2 browser: converting local midnight to UTC before
    // taking end-of-day used to move the window onto the previous UTC day.
    process.env.TZ = "Europe/Berlin";
  });

  afterAll(() => {
    process.env.TZ = originalTimezone;
  });

  it("covers the whole selected local range when looking up the clicked request", async () => {
    renderViewer("2026-07-22", "2026-07-22");

    fireEvent.click(screen.getByText("hello"));

    await waitFor(() => expect(uiSpendLogsCall).toHaveBeenCalled());

    const { start_date, end_date, params } = vi.mocked(uiSpendLogsCall).mock.calls[0][0];
    expect(params?.request_id).toBe("chatcmpl-1");

    const logTimestamp = moment.utc("2026-07-22T09:38:46.397Z");
    expect(moment.utc(start_date, "YYYY-MM-DD HH:mm:ss").isSameOrBefore(logTimestamp)).toBe(true);
    expect(moment.utc(end_date, "YYYY-MM-DD HH:mm:ss").isSameOrAfter(logTimestamp)).toBe(true);
  });
});
