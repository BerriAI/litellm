import { render, screen } from "@testing-library/react";
import React from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { STACKED_USAGE_PALETTE, StackedUsageChart, StackedUsageTooltipRow, totalsFitPlot } from "./StackedUsageChart";

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

const usd = (value: number) => `$${value.toFixed(2)}`;

describe("StackedUsageChart", () => {
  it("stacks one bar per series per bucket in the shared palette, in series order", () => {
    const { container } = render(
      <StackedUsageChart
        data={[
          { day: "Oct 1", a: 1, b: 2 },
          { day: "Oct 2", a: 3, b: 4 },
        ]}
        series={["a", "b"]}
        xKey="day"
        format={usd}
      />,
    );
    const fills = Array.from(container.querySelectorAll("path.recharts-rectangle")).map((bar) =>
      bar.getAttribute("fill"),
    );
    expect(fills).toHaveLength(4);
    expect(new Set(fills)).toEqual(new Set([STACKED_USAGE_PALETTE[0], STACKED_USAGE_PALETTE[1]]));
  });

  it("lets a caller override a series color, e.g. a slate Other bucket", () => {
    const { container } = render(
      <StackedUsageChart
        data={[{ day: "Oct 1", a: 1, Other: 2 }]}
        series={["a", "Other"]}
        colors={[STACKED_USAGE_PALETTE[0], "#94a3b8"]}
        xKey="day"
        format={usd}
      />,
    );
    const fills = Array.from(container.querySelectorAll("path.recharts-rectangle")).map((bar) =>
      bar.getAttribute("fill"),
    );
    expect(fills).toContain("#94a3b8");
  });
});

describe("StackedUsageChart bar totals", () => {
  afterEach(() => vi.unstubAllGlobals());

  const data = [
    { day: "2026-10", a: 1, b: 2 },
    { day: "2026-11", a: 4, b: 0 },
  ];
  const totals = new Map([
    ["2026-10", 13],
    ["2026-11", 17],
  ]);

  it("prints each bucket's total above its bar, including one whose top series is zero", () => {
    render(
      <StackedUsageChart
        data={data}
        series={["a", "b"]}
        xKey="day"
        format={usd}
        totalFor={(day) => totals.get(day)}
        showTotals
      />,
    );
    expect(screen.getByText("$13.00")).toBeInTheDocument();
    expect(screen.getByText("$17.00")).toBeInTheDocument();
  });

  it("drops totals that would collide in a narrow chart even when the window is wide", () => {
    vi.stubGlobal("innerWidth", 1440);
    vi.stubGlobal("ResizeObserver", chartContainerOfWidth(140));
    render(
      <StackedUsageChart
        data={data}
        series={["a", "b"]}
        xKey="day"
        format={usd}
        totalFor={(day) => totals.get(day)}
        showTotals
      />,
    );
    expect(screen.queryByText("$13.00")).not.toBeInTheDocument();
    expect(screen.queryByText("$17.00")).not.toBeInTheDocument();
  });

  it("drops totals when a roomy chart holds more buckets than their labels fit", () => {
    const month = Array.from({ length: 31 }, (_, index) => ({ day: `2026-10-${index + 1}`, a: 1, b: 2 }));
    render(
      <StackedUsageChart data={month} series={["a", "b"]} xKey="day" format={usd} totalFor={() => 13} showTotals />,
    );
    expect(screen.queryByText("$13.00")).not.toBeInTheDocument();
  });

  it("prints no totals on a log scale, where a stack top is not the sum", () => {
    render(
      <StackedUsageChart
        data={data}
        series={["a", "b"]}
        xKey="day"
        format={usd}
        totalFor={(day) => totals.get(day)}
        showTotals
        scale="log"
      />,
    );
    expect(screen.queryByText("$13.00")).not.toBeInTheDocument();
  });
});

describe("totalsFitPlot", () => {
  it("fits labels only when the widest one has room in every bucket's slot", () => {
    expect(totalsFitPlot(600, ["$1.2K", "$624.1M", "$3"])).toBe(true);
    expect(totalsFitPlot(150, ["$1.2K", "$624.1M", "$3"])).toBe(false);
    expect(totalsFitPlot(600, [])).toBe(false);
  });
});

describe("StackedUsageTooltipRow", () => {
  it("formats the value and keeps it apart from a long series name", () => {
    render(<StackedUsageTooltipRow value={0.091} name="cerebras/gpt-oss-120b" color="#a855f7" format={usd} />);
    expect(screen.getByText("cerebras/gpt-oss-120b")).toBeInTheDocument();
    expect(screen.getByText("$0.09")).toBeInTheDocument();
    expect(screen.queryByText("0.091")).not.toBeInTheDocument();
  });

  it("renders nothing for a zero value, so a day lists only what it used", () => {
    const { container } = render(<StackedUsageTooltipRow value={0} name="gpt-4.1-mini" color="#3b82f6" format={usd} />);
    expect(container).toBeEmptyDOMElement();
  });
});
