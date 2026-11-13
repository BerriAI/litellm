import { render, screen } from "@testing-library/react";
import React from "react";
import { describe, expect, it } from "vitest";
import { STACKED_USAGE_PALETTE, StackedUsageChart, StackedUsageTooltipRow } from "./StackedUsageChart";

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
  const data = [
    { day: "2026-10", a: 1, b: 2 },
    { day: "2026-11", a: 4, b: 0 },
  ];
  const totals = new Map([
    ["2026-10", 13],
    ["2026-11", 17],
  ]);

  it("prints each bucket's total above its bar, including one whose top series is zero", () => {
    const { container } = render(
      <StackedUsageChart
        data={data}
        series={["a", "b"]}
        xKey="day"
        format={usd}
        totalFor={(day) => totals.get(day)}
        showTotals
      />,
    );
    const labels = Array.from(container.querySelectorAll("text")).map((node) => node.textContent);
    expect(labels).toContain("$13.00");
    expect(labels).toContain("$17.00");
  });

  it("prints no totals on a log scale, where a stack top is not the sum", () => {
    const { container } = render(
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
    const labels = Array.from(container.querySelectorAll("text")).map((node) => node.textContent);
    expect(labels).not.toContain("$13.00");
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
