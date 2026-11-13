import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { DailyData, SpendMetrics } from "@/components/UsagePage/types";
import TrendControls from "./TrendControls";
import { useTrendSelection } from "./trendSelection";
import UsageTrendPanel from "./UsageTrendPanel";
import type { DayRange } from "./usageTrendData";

const metricsOf = (spend: number, tokens: number): SpendMetrics => ({
  spend,
  prompt_tokens: tokens,
  completion_tokens: 0,
  total_tokens: tokens,
  api_requests: 1,
  successful_requests: 1,
  failed_requests: 0,
  cache_read_input_tokens: 0,
  cache_creation_input_tokens: 0,
});

const day = (date: string, models: Record<string, readonly [number, number]>): DailyData => {
  const entries = Object.entries(models);
  const spend = entries.reduce((sum, [, [value]]) => sum + value, 0);
  const tokens = entries.reduce((sum, [, [, value]]) => sum + value, 0);
  return {
    date,
    metrics: metricsOf(spend, tokens),
    breakdown: {
      models: {},
      model_groups: Object.fromEntries(
        entries.map(([name, [value, count]]) => [
          name,
          { metrics: metricsOf(value, count), metadata: {}, api_key_breakdown: {} },
        ]),
      ),
      mcp_servers: {},
      providers: {},
      api_keys: {},
      entities: {},
      endpoints: {},
    },
  };
};

const chartContainerOfWidth = (width: number) =>
  class {
    private readonly callback: ResizeObserverCallback;
    constructor(callback: ResizeObserverCallback) {
      this.callback = callback;
    }
    observe(target: Element) {
      const box = { inlineSize: width, blockSize: 400 };
      const entry = {
        target,
        contentRect: new DOMRect(0, 0, width, 400),
        borderBoxSize: [box],
        contentBoxSize: [box],
        devicePixelContentBoxSize: [box],
      } satisfies ResizeObserverEntry;
      this.callback([entry], this as unknown as ResizeObserver);
    }
    unobserve() {}
    disconnect() {}
  };

function Trend({ results, range, compact }: { results: readonly DailyData[]; range: DayRange; compact?: boolean }) {
  const state = useTrendSelection(range, results);
  return (
    <UsageTrendPanel
      results={results}
      range={range}
      loading={false}
      failed={false}
      selection={state.selection}
      compact={compact}
      action={<TrendControls state={state} />}
    />
  );
}

const quarter = { start: "2026-08-01", end: "2026-10-31" };
const quarterly = [
  day("2026-08-10", { alpha: [777, 7777] }),
  day("2026-09-10", { alpha: [777, 7777] }),
  day("2026-10-10", { alpha: [777, 7777] }),
];

describe("UsageTrendPanel", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("charts one metric at a time, tokens first, and retitles the chart as the metric and bucket change", async () => {
    const user = userEvent.setup();
    render(<Trend results={quarterly} range={quarter} />);

    expect(screen.getByRole("group", { name: "Monthly tokens by model" })).toBeInTheDocument();
    expect(screen.getByText("Monthly tokens by model, Aug 1 – Oct 31")).toBeInTheDocument();
    expect(screen.getAllByText("7.8K")).toHaveLength(3);
    expect(screen.queryByText("$777")).not.toBeInTheDocument();
    expect(screen.getByRole("list", { name: "Models" })).toHaveTextContent("alpha");

    await user.click(screen.getByRole("radio", { name: "Spend" }));
    expect(screen.getByRole("group", { name: "Monthly spend by model" })).toBeInTheDocument();
    expect(screen.getAllByText("$777")).toHaveLength(3);
    expect(screen.queryByText("7.8K")).not.toBeInTheDocument();

    await user.click(screen.getByRole("radio", { name: "Weekly" }));
    expect(screen.getByText("Weekly spend by model, Aug 1 – Oct 31")).toBeInTheDocument();
  });

  it("keeps a narrow compact chart free of crowded totals and leaves exact figures in the table", () => {
    vi.stubGlobal("innerWidth", 1440);
    vi.stubGlobal("ResizeObserver", chartContainerOfWidth(150));
    render(<Trend results={quarterly} range={quarter} compact />);

    expect(screen.queryByText("7.8K")).not.toBeInTheDocument();
    const table = screen.getByRole("region", { name: "Contribution by model" });
    expect(within(table).getByRole("row", { name: /alpha/ })).toHaveTextContent("$2,331.00");
  });

  it("labels clipped months and weeks by the days they cover", async () => {
    const user = userEvent.setup();
    const results = [day("2026-10-20", { alpha: [5, 50] }), day("2026-11-05", { alpha: [7, 70] })];
    render(<Trend results={results} range={{ start: "2026-10-15", end: "2026-11-12" }} />);
    const bucketSize = screen.getByRole("radiogroup", { name: "Bucket size" });

    await user.click(within(bucketSize).getByRole("radio", { name: "Monthly" }));
    expect(screen.getAllByText("Oct 15–31").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Nov 1–12").length).toBeGreaterThan(0);
    expect(
      screen.getByText(/only the days inside the selected range: Oct 15 – Oct 31, Nov 1 – Nov 12\./),
    ).toBeVisible();

    await user.click(within(bucketSize).getByRole("radio", { name: "Weekly" }));
    expect(screen.getAllByText("Oct 22").length).toBeGreaterThan(0);
    expect(screen.getByText(/only the days inside the selected range: Nov 12, 2026\./)).toBeVisible();

    await user.click(within(bucketSize).getByRole("radio", { name: "Daily" }));
    expect(screen.queryByText(/only the days inside the selected range/)).not.toBeInTheDocument();
  });

  it("names spend without tokens and tokens without recorded spend instead of a rate", () => {
    const results = [day("2026-10-03", { images: [0.4, 0], bundled: [0, 5000] })];
    render(<Trend results={results} range={{ start: "2026-10-01", end: "2026-10-31" }} />);
    const table = screen.getByRole("region", { name: "Contribution by model" });

    expect(within(table).getByRole("row", { name: /images/ })).toHaveTextContent("Spend without tokens");
    expect(within(table).getByRole("row", { name: /bundled/ })).toHaveTextContent("No recorded spend");
  });
});
