import { fireEvent, render, screen } from "@testing-library/react";
import React from "react";
import { describe, expect, it, vi } from "vitest";
import { BarChart } from "./bar_chart";

const data = [
  { date: "2026-03-01", passed: 10, blocked: 2 },
  { date: "2026-03-02", passed: 15, blocked: 1 },
];

describe("BarChart", () => {
  it("renders one bar series per category with the mapped tremor colors", () => {
    const { container } = render(
      <BarChart data={data} index="date" categories={["passed", "blocked"]} colors={["green", "red"]} />,
    );

    const rectangles = Array.from(container.querySelectorAll("path.recharts-rectangle"));
    expect(rectangles).toHaveLength(4);
    const fills = new Set(rectangles.map((rect) => rect.getAttribute("fill")));
    expect(fills).toEqual(new Set(["var(--color-green-500, #22c55e)", "var(--color-red-500, #ef4444)"]));
  });

  it("falls back to the tremor default color cycle when no colors are passed", () => {
    const { container } = render(<BarChart data={data} index="date" categories={["passed", "blocked"]} />);

    const fills = new Set(
      Array.from(container.querySelectorAll("path.recharts-rectangle")).map((rect) => rect.getAttribute("fill")),
    );
    expect(fills).toEqual(new Set(["var(--color-blue-500, #3b82f6)", "var(--color-cyan-500, #06b6d4)"]));
  });

  it("fires onValueChange with the datum and clicked category", () => {
    const onValueChange = vi.fn();
    const { container } = render(
      <BarChart
        data={data}
        index="date"
        categories={["passed", "blocked"]}
        colors={["green", "red"]}
        onValueChange={onValueChange}
      />,
    );

    const firstRect = container.querySelector("path.recharts-rectangle");
    expect(firstRect).not.toBeNull();
    fireEvent.click(firstRect!);

    expect(onValueChange).toHaveBeenCalledTimes(1);
    const expectedClickItem = {
      date: "2026-03-01",
      passed: 10,
      blocked: 2,
      categoryClicked: "passed",
    };
    expect(onValueChange).toHaveBeenCalledWith(expectedClickItem);
  });

  it("renders category labels on the y axis in vertical layout", () => {
    render(
      <BarChart
        data={[
          { key: "alpha", spend: 12 },
          { key: "beta", spend: 7 },
        ]}
        index="key"
        categories={["spend"]}
        colors={["cyan"]}
        layout="vertical"
        yAxisWidth={120}
      />,
    );

    expect(screen.getAllByText("alpha").length).toBeGreaterThan(0);
    expect(screen.getAllByText("beta").length).toBeGreaterThan(0);
  });

  it("applies valueFormatter to the value axis ticks", () => {
    render(
      <BarChart
        data={data}
        index="date"
        categories={["passed"]}
        colors={["green"]}
        valueFormatter={(v) => `${v} req`}
      />,
    );

    expect(screen.getAllByText(/ req$/).length).toBeGreaterThan(0);
  });

  it("renders a legend by default, matching tremor, and hides it when showLegend is false", () => {
    const { container, rerender } = render(
      <BarChart data={data} index="date" categories={["passed"]} colors={["green"]} />,
    );
    expect(screen.getByText("passed")).toBeInTheDocument();
    expect(container.querySelector(".recharts-legend-wrapper")).not.toBeNull();

    rerender(<BarChart data={data} index="date" categories={["passed"]} colors={["green"]} showLegend={false} />);
    expect(screen.queryByText("passed")).not.toBeInTheDocument();
  });

  it("emits no per-chart style tag; colors flow through fills, not CSS vars", () => {
    const { container } = render(
      <BarChart data={data} index="date" categories={["passed", "blocked"]} colors={["green", "red"]} />,
    );
    expect(container.querySelector("style")).toBeNull();
  });

  it("stacks bars into a single column per index when stack is set", () => {
    const { container } = render(
      <BarChart data={data} index="date" categories={["passed", "blocked"]} colors={["green", "red"]} stack={true} />,
    );

    const xPositions = Array.from(container.querySelectorAll("path.recharts-rectangle")).map(
      (rect) => rect.getAttribute("d")?.split(",")[0],
    );
    expect(new Set(xPositions).size).toBe(2);
  });
});
