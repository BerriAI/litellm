import Papa from "papaparse";
import { beforeEach, expect, it, vi } from "vitest";
import { downloadCsv } from "@/app/(dashboard)/usage/_components/components/EntityUsage/teamUserSpend";
import { uiSpendLogsCall } from "../../networking";
import type { LogEntry } from "../types";
import { exportLogsCsv, LOGS_EXPORT_LIMIT } from "./logsCsvExport";

vi.mock("../../networking", () => ({ uiSpendLogsCall: vi.fn() }));
vi.mock("@/app/(dashboard)/usage/_components/components/EntityUsage/teamUserSpend", () => ({ downloadCsv: vi.fn() }));

const options = {
  accessToken: "test-token",
  start_date: "2026-01-01",
  end_date: "2026-01-02",
  params: { search: "needle", team_id: "team-1", sort_by: "spend", sort_order: "asc" as const, group_by_session: true },
};
const log = (id: string): LogEntry => ({
  request_id: id,
  api_key: "hash",
  team_id: "team-1",
  model: "=FORMULA()",
  model_id: "model-1",
  call_type: "completion",
  spend: 0.01,
  total_tokens: 3,
  prompt_tokens: 1,
  completion_tokens: 2,
  startTime: "2026-01-01T00:00:00Z",
  endTime: "2026-01-01T00:00:01Z",
  cache_hit: "false",
  end_user: "=1+\n1",
  messages: [],
  response: {},
});

beforeEach(() => vi.resetAllMocks());

it("exports every page with frozen filters and escapes spreadsheet formulas", async () => {
  vi.mocked(uiSpendLogsCall)
    .mockResolvedValueOnce({ data: [log("first")], total_pages: 2 })
    .mockResolvedValueOnce({ data: [log("second")], total_pages: 2 });
  await expect(exportLogsCsv(options)).resolves.toEqual({ count: 2, capped: false });
  expect(uiSpendLogsCall).toHaveBeenCalledTimes(2);
  for (const page of [1, 2]) {
    const expectedOptions = {
      ...options,
      page,
      page_size: 100,
      params: { ...options.params, group_by_session: false, session_cursor: undefined },
    };
    expect(uiSpendLogsCall).toHaveBeenNthCalledWith(page, expectedOptions);
  }
  const csv = vi.mocked(downloadCsv).mock.calls[0][0];
  const rows = Papa.parse<Record<string, string>>(csv, { header: true }).data;
  expect(rows.map((row) => row["Request ID"])).toEqual(["first", "second"]);
  expect(rows.map((row) => row.Model)).toEqual(["'=FORMULA()", "'=FORMULA()"]);
  expect(rows.map((row) => row["End User"])).toEqual(["'=1+\n1", "'=1+\n1"]);
});

it("does not download an empty result", async () => {
  vi.mocked(uiSpendLogsCall).mockResolvedValue({ data: [], total_pages: 0 });
  await expect(exportLogsCsv(options)).resolves.toEqual({ count: 0, capped: false });
  expect(downloadCsv).not.toHaveBeenCalled();
});

it("does not download partial results after a failed page", async () => {
  vi.mocked(uiSpendLogsCall)
    .mockResolvedValueOnce({ data: [log("first")], total_pages: 2 })
    .mockRejectedValueOnce(new Error("Unavailable"));
  await expect(exportLogsCsv(options)).rejects.toThrow("Unavailable");
  expect(downloadCsv).not.toHaveBeenCalled();
});

it("stops fetching at the row cap", async () => {
  vi.mocked(uiSpendLogsCall).mockResolvedValue({
    data: Array.from({ length: 100 }, (_, index) => log(String(index))),
    total_pages: 200,
  });
  await expect(exportLogsCsv(options)).resolves.toEqual({ count: LOGS_EXPORT_LIMIT, capped: true });
  expect(uiSpendLogsCall).toHaveBeenCalledTimes(LOGS_EXPORT_LIMIT / 100);
  expect(Papa.parse(vi.mocked(downloadCsv).mock.calls[0][0], { header: true }).data).toHaveLength(LOGS_EXPORT_LIMIT);
});
