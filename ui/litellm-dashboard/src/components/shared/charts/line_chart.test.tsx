import { render, screen } from "@testing-library/react";
import React from "react";
import { describe, expect, it } from "vitest";
import { LineChart } from "./line_chart";

const data = [
  { date: "Jun 1", "/chat/completions": 10, "/embeddings": 4 },
  { date: "Jun 2", "/chat/completions": 15, "/embeddings": 6 },
  { date: "Jun 3", "/chat/completions": 12, "/embeddings": 9 },
];

describe("LineChart", () => {
  it("renders one line per category with the mapped tremor stroke colors", () => {
    const { container } = render(
      <LineChart
        data={data}
        index="date"
        categories={["/chat/completions", "/embeddings"]}
        colors={["blue", "cyan"]}
      />,
    );

    const curves = Array.from(container.querySelectorAll("path.recharts-line-curve"));
    expect(curves).toHaveLength(2);
    expect(curves.map((curve) => curve.getAttribute("stroke"))).toEqual([
      "var(--color-blue-500, #3b82f6)",
      "var(--color-cyan-500, #06b6d4)",
    ]);
  });

  it("falls back to the tremor default color cycle when no colors are passed", () => {
    const { container } = render(
      <LineChart data={data} index="date" categories={["/chat/completions", "/embeddings"]} />,
    );

    const strokes = Array.from(container.querySelectorAll("path.recharts-line-curve")).map((curve) =>
      curve.getAttribute("stroke"),
    );
    expect(strokes).toEqual(["var(--color-blue-500, #3b82f6)", "var(--color-cyan-500, #06b6d4)"]);
  });

  it("applies valueFormatter to the value axis ticks", () => {
    render(
      <LineChart
        data={data}
        index="date"
        categories={["/chat/completions"]}
        colors={["blue"]}
        valueFormatter={(v) => `${v} req`}
      />,
    );

    expect(screen.getAllByText(/ req$/).length).toBeGreaterThan(0);
  });

  it("renders a legend by default, matching tremor, and hides it when showLegend is false", () => {
    const { container, rerender } = render(
      <LineChart data={data} index="date" categories={["/chat/completions"]} colors={["blue"]} />,
    );
    expect(screen.getByText("/chat/completions")).toBeInTheDocument();
    expect(container.querySelector(".recharts-legend-wrapper")).not.toBeNull();

    rerender(
      <LineChart data={data} index="date" categories={["/chat/completions"]} colors={["blue"]} showLegend={false} />,
    );
    expect(screen.queryByText("/chat/completions")).not.toBeInTheDocument();
  });

  it("draws straight segments by default and curved segments for curveType natural", () => {
    const { container: linear } = render(
      <LineChart data={data} index="date" categories={["/chat/completions"]} colors={["blue"]} />,
    );
    const { container: natural } = render(
      <LineChart data={data} index="date" categories={["/chat/completions"]} colors={["blue"]} curveType="natural" />,
    );

    const linearPath = linear.querySelector("path.recharts-line-curve")?.getAttribute("d") ?? "";
    const naturalPath = natural.querySelector("path.recharts-line-curve")?.getAttribute("d") ?? "";
    expect(linearPath).not.toContain("C");
    expect(naturalPath).toContain("C");
  });

  it("bridges gaps over null values only when connectNulls is set", () => {
    const gappedData = [
      { date: "Jun 1", "/chat/completions": 10 },
      { date: "Jun 2", "/chat/completions": null },
      { date: "Jun 3", "/chat/completions": 12 },
      { date: "Jun 4", "/chat/completions": 15 },
    ];

    const { container: broken } = render(
      <LineChart data={gappedData} index="date" categories={["/chat/completions"]} colors={["blue"]} />,
    );
    const { container: bridged } = render(
      <LineChart data={gappedData} index="date" categories={["/chat/completions"]} colors={["blue"]} connectNulls />,
    );

    const brokenPath = broken.querySelector("path.recharts-line-curve")?.getAttribute("d") ?? "";
    const bridgedPath = bridged.querySelector("path.recharts-line-curve")?.getAttribute("d") ?? "";
    expect((brokenPath.match(/M/g) ?? []).length).toBeGreaterThan(1);
    expect((bridgedPath.match(/M/g) ?? []).length).toBe(1);
  });

  it("renders an empty chart without lines when there are no categories", () => {
    const { container } = render(<LineChart data={[]} index="date" categories={[]} />);

    expect(container.querySelector("[data-slot='chart']")).not.toBeNull();
    expect(container.querySelectorAll("path.recharts-line-curve")).toHaveLength(0);
  });

  it("emits no per-chart style tag; colors flow through strokes, not CSS vars", () => {
    const { container } = render(
      <LineChart data={data} index="date" categories={["/chat/completions"]} colors={["blue"]} />,
    );
    expect(container.querySelector("style")).toBeNull();
  });
});
