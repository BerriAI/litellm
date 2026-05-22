import React from "react";
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useLogFilterLogic } from "../../src/components/view_logs/log_filter_logic";

// Minimal mocks to avoid real network during hook init
vi.mock("../../src/components/key_team_helpers/filter_helpers", () => ({
  fetchAllKeyAliases: vi.fn().mockResolvedValue([]),
  fetchAllTeams: vi.fn().mockResolvedValue([]),
}));

const createQueryClient = () =>
  new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });

function Harness({ logs }: { logs: any }) {
  const { filteredLogs } = useLogFilterLogic({
    logs,
    accessToken: "token",
    startTime: "2025-01-01 00:00:00",
    endTime: "2025-01-02 00:00:00",
    pageSize: 50,
    isCustomDate: true,
    setCurrentPage: () => {},
    userID: "user-1",
    userRole: "admin",
  });

  return <div data-testid="count">{filteredLogs.data.length}</div>;
}

describe("useLogFilterLogic (minimal)", () => {
  it("useLogFilterLogic minimal: updates filteredLogs when logs change", async () => {
    const qc = createQueryClient();
    const logsA = { data: [{ request_id: "a" }], total: 1, page: 1, page_size: 50, total_pages: 1 };
    const logsB = {
      data: [{ request_id: "a" }, { request_id: "b" }],
      total: 2,
      page: 1,
      page_size: 50,
      total_pages: 1,
    };

    const { rerender } = render(
      <QueryClientProvider client={qc}>
        <Harness logs={logsA} />
      </QueryClientProvider>,
    );

    expect(await screen.findByTestId("count")).toHaveTextContent("1");

    rerender(
      <QueryClientProvider client={qc}>
        <Harness logs={logsB} />
      </QueryClientProvider>,
    );

    expect(await screen.findByTestId("count")).toHaveTextContent("2");
  });
});
