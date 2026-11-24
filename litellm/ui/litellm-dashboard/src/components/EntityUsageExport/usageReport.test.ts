// @vitest-environment jsdom
import type { TFunction } from "i18next";
import { afterEach, describe, expect, it, vi } from "vitest";
import i18n from "@/i18n/i18n";
import {
  USAGE_REPORT_MAX_ROWS,
  buildUsageReportHtml,
  parseUsageReportRows,
  printUsageReport,
  type UsageReportRow,
} from "./usageReport";

const t = i18n.getFixedT("en") as TFunction;

const row = (overrides: Partial<UsageReportRow> = {}): UsageReportRow => ({
  date: "2025-10-02",
  entity_id: "team-a",
  entity_alias: null,
  api_key: "hashed-key",
  key_alias: null,
  user_id: null,
  user_email: null,
  model: null,
  spend: 0,
  prompt_tokens: 0,
  completion_tokens: 0,
  api_requests: 0,
  successful_requests: 0,
  failed_requests: 0,
  ...overrides,
});

type ReportInput = Parameters<typeof buildUsageReportHtml>[0];

const baseInput: ReportInput = {
  title: "team_usage_daily_2025-10-01_2025-10-14",
  heading: "Export Team Usage",
  entityType: "team",
  exportType: "daily",
  rows: [row()],
  dateRange: { from: new Date(2025, 9, 1), to: new Date(2025, 9, 14) },
  selectedFilters: [],
  generatedAt: new Date(2025, 9, 15),
  t,
};

const render = (overrides: Partial<ReportInput> = {}): Document => {
  const input = { ...baseInput, ...overrides };
  return new DOMParser().parseFromString(buildUsageReportHtml(input), "text/html");
};

const headers = (doc: Document) => [...doc.querySelectorAll("thead th")].map((cell) => cell.textContent);
const bodyRows = (doc: Document) =>
  [...doc.querySelectorAll("tbody tr")].map((tr) => [...tr.querySelectorAll("td")].map((cell) => cell.textContent));

describe("buildUsageReportHtml", () => {
  it.each([
    ["daily", [], "Team A"],
    ["daily_with_keys", ["Key"], "prod-key"],
    ["daily_with_models", ["Model"], "gpt-4o"],
    ["daily_with_users", ["User"], "a@example.com"],
  ] as const)("adds the %s breakdown column", (exportType, extraHeader, expectedCell) => {
    const labelled = { entity_alias: "Team A", key_alias: "prod-key", model: "gpt-4o", user_email: "a@example.com" };
    const doc = render({ exportType, rows: [row(labelled)] });

    expect(headers(doc)).toEqual([
      "Date",
      "Team",
      ...extraHeader,
      "Spend",
      "Requests",
      "Successful",
      "Failed",
      "Prompt tokens",
      "Completion tokens",
    ]);
    expect(bodyRows(doc)[0]).toContain(expectedCell);
  });

  it.each([
    ["daily_with_keys", row({ api_key: "hashed-key" }), "hashed-key"],
    ["daily_with_users", row({ user_id: "user-1" }), "user-1"],
    ["daily_with_models", row(), "-"],
    ["daily", row({ entity_alias: null, entity_id: "team-a" }), "team-a"],
  ] as const)("falls back to the raw id when the %s label is missing", (exportType, input, expectedCell) => {
    expect(bodyRows(render({ exportType, rows: [input] }))[0]).toContain(expectedCell);
  });

  it("caps the table but keeps totals over every row", () => {
    const rows = Array.from({ length: USAGE_REPORT_MAX_ROWS + 5 }, () => row({ spend: 1, api_requests: 2 }));
    const doc = render({ rows });

    expect(bodyRows(doc)).toHaveLength(USAGE_REPORT_MAX_ROWS);
    expect(doc.querySelector(".note")?.textContent).toBe(
      "Showing the first 1,000 of 1,005 rows. Totals cover all rows. Export as CSV for the full breakdown.",
    );
    const summary = doc.querySelector('[data-testid="usage-report-summary"]')?.textContent;
    expect(summary).toContain("Total spend$1,005.00");
    expect(summary).toContain("Requests2,010");
  });

  it("omits the truncation note when every row fits", () => {
    expect(
      render({ rows: Array.from({ length: USAGE_REPORT_MAX_ROWS }, () => row()) }).querySelector(".note"),
    ).toBeNull();
  });

  it("sums every counter into the summary", () => {
    const first = {
      spend: 0.5,
      api_requests: 3,
      successful_requests: 2,
      failed_requests: 1,
      prompt_tokens: 10,
      completion_tokens: 5,
    };
    const second = { spend: 0.25, api_requests: 1, successful_requests: 1, prompt_tokens: 20, completion_tokens: 7 };
    const doc = render({ rows: [row(first), row(second)] });

    const cards = [...doc.querySelectorAll('[data-testid="usage-report-summary"] > div')].map((card) => [
      card.querySelector(".label")?.textContent,
      card.querySelector(".value")?.textContent,
    ]);
    expect(cards).toEqual([
      ["Total spend", "$0.75"],
      ["Requests", "4"],
      ["Successful requests", "3"],
      ["Failed requests", "1"],
      ["Total tokens", "42"],
      ["Prompt tokens", "30"],
      ["Completion tokens", "12"],
    ]);
  });

  it("keeps sub-cent spend visible", () => {
    expect(bodyRows(render({ rows: [row({ spend: 0.000123 })] }))[0]).toContain("$0.000123");
  });

  it("shows the title, date range and filters", () => {
    const doc = render({ selectedFilters: ["team-a", "team-b"] });

    expect(doc.title).toBe("team_usage_daily_2025-10-01_2025-10-14");
    expect(doc.querySelector("h1")?.textContent).toBe("Export Team Usage");
    const meta = doc.querySelector(".meta")?.textContent;
    expect(meta).toContain(
      `Date range: ${new Date(2025, 9, 1).toLocaleDateString()} - ${new Date(2025, 9, 14).toLocaleDateString()}`,
    );
    expect(meta).toContain("Filters: team-a, team-b");
  });

  it("leaves out the filters line when nothing is filtered", () => {
    expect(render().querySelector(".meta")?.textContent).not.toContain("Filters");
  });

  it("renders an empty state when there are no rows", () => {
    expect(bodyRows(render({ rows: [] }))).toEqual([["No usage in this date range"]]);
  });

  it("escapes values coming from the export so they render as text", () => {
    const doc = render({
      rows: [row({ entity_alias: '<img src=x onerror="alert(1)">' })],
      selectedFilters: ["<script>alert(1)</script>"],
    });

    expect(doc.querySelector("img")).toBeNull();
    expect(doc.querySelector("script")).toBeNull();
    expect(bodyRows(doc)[0][1]).toBe('<img src=x onerror="alert(1)">');
  });
});

describe("parseUsageReportRows", () => {
  const blobOf = (text: string) => Object.assign(new Blob([text]), { text: async () => text });

  it("reads the JSON export rows", async () => {
    await expect(
      parseUsageReportRows(blobOf(JSON.stringify([{ ...row(), flat_cost: 0, key_metadata: {} }]))),
    ).resolves.toEqual([row()]);
  });

  it("rejects a payload that is not an array of export rows", async () => {
    await expect(parseUsageReportRows(blobOf(JSON.stringify({ detail: "error" })))).rejects.toThrow();
  });
});

describe("printUsageReport", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    document.querySelectorAll("iframe").forEach((frame) => frame.remove());
  });

  const capturePrints = () => {
    const prints: ReturnType<typeof vi.fn>[] = [];
    const append = document.body.appendChild.bind(document.body);
    vi.spyOn(document.body, "appendChild").mockImplementation(<T extends Node>(node: T): T => {
      const appended = append(node);
      if (node instanceof HTMLIFrameElement && node.contentWindow) {
        const print = vi.fn();
        node.contentWindow.print = print;
        prints.push(print);
      }
      return appended;
    });
    return prints;
  };

  it("names the saved file after the report while the print dialog is open", () => {
    document.title = "LiteLLM Dashboard";
    const titles: string[] = [];
    const append = document.body.appendChild;
    vi.spyOn(document.body, "appendChild").mockImplementation(<T extends Node>(node: T): T => {
      const appended = append.call(document.body, node) as T;
      if (node instanceof HTMLIFrameElement && node.contentWindow) {
        node.contentWindow.print = () => titles.push(document.title);
      }
      return appended;
    });

    printUsageReport("<title>report</title>", "Usage report", "team_usage_daily_2025-10-01_2025-10-31");

    expect(titles).toEqual(["team_usage_daily_2025-10-01_2025-10-31"]);
    expect(document.title).toBe("LiteLLM Dashboard");
  });

  it("prints the report from a hidden frame", () => {
    const prints = capturePrints();

    printUsageReport(
      "<html><head><title>report-title</title></head><body><h1>Hello</h1></body></html>",
      "Usage report",
      "team_usage_daily",
    );

    const frame = document.querySelector<HTMLIFrameElement>('iframe[title="Usage report"]');
    expect(frame?.contentDocument?.title).toBe("report-title");
    expect(frame?.contentDocument?.querySelector("h1")?.textContent).toBe("Hello");
    expect(prints).toHaveLength(1);
    expect(prints[0]).toHaveBeenCalledOnce();
  });

  it("replaces the frame from a previous export", () => {
    capturePrints();

    printUsageReport("<title>first</title>", "Usage report", "first");
    printUsageReport("<title>second</title>", "Usage report", "second");

    const frames = document.querySelectorAll<HTMLIFrameElement>("iframe");
    expect(frames).toHaveLength(1);
    expect(frames[0].contentDocument?.title).toBe("second");
  });
});
