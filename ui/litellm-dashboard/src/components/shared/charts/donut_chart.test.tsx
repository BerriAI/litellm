import { render } from "@testing-library/react";
import React from "react";
import { describe, expect, it } from "vitest";
import { DonutChart } from "./donut_chart";

const data = [
  { provider: "openai", spend: 40 },
  { provider: "anthropic", spend: 30 },
  { provider: "bedrock", spend: 20 },
];

describe("DonutChart", () => {
  it("renders one sector per datum, cycling the given colors", () => {
    const { container } = render(
      <DonutChart data={data} index="provider" category="spend" colors={["cyan", "blue"]} />,
    );

    const sectors = Array.from(container.querySelectorAll(".recharts-pie-sector path"));
    expect(sectors).toHaveLength(3);
    expect(sectors.map((sector) => sector.getAttribute("fill"))).toEqual([
      "var(--color-cyan-500, #06b6d4)",
      "var(--color-blue-500, #3b82f6)",
      "var(--color-cyan-500, #06b6d4)",
    ]);
  });

  it("renders a full pie when variant is pie and a hollow donut otherwise", () => {
    const { container: donut } = render(<DonutChart data={data} index="provider" category="spend" colors={["cyan"]} />);
    const { container: pie } = render(
      <DonutChart data={data} index="provider" category="spend" colors={["cyan"]} variant="pie" />,
    );

    const donutPath = donut.querySelector(".recharts-pie-sector path")?.getAttribute("d") ?? "";
    const piePath = pie.querySelector(".recharts-pie-sector path")?.getAttribute("d") ?? "";
    expect(donutPath).not.toEqual(piePath);
    expect((donutPath.match(/A/g) ?? []).length).toBeGreaterThan((piePath.match(/A/g) ?? []).length);
  });

  it("hides the center label by default and shows the formatted total when showLabel is set", () => {
    const { container, rerender } = render(
      <DonutChart
        data={data}
        index="provider"
        category="spend"
        colors={["cyan"]}
        valueFormatter={(value) => `$${value.toFixed(2)}`}
      />,
    );
    expect(container.querySelector("text.fill-foreground")).toBeNull();

    rerender(
      <DonutChart
        data={data}
        index="provider"
        category="spend"
        colors={["cyan"]}
        valueFormatter={(value) => `$${value.toFixed(2)}`}
        showLabel
      />,
    );
    expect(container.querySelector("text.fill-foreground")?.textContent).toBe("$90.00");
  });

  it("never invokes the valueFormatter for the center label unless it is shown", () => {
    const formatterCalls: number[] = [];
    render(
      <DonutChart
        data={data}
        index="provider"
        category="spend"
        colors={["cyan"]}
        showTooltip={false}
        valueFormatter={(value) => {
          formatterCalls.push(value);
          return `$${value.toFixed(2)}`;
        }}
      />,
    );
    expect(formatterCalls).toEqual([]);
  });

  it("prefers an explicit label over the computed total", () => {
    const { container } = render(
      <DonutChart data={data} index="provider" category="spend" colors={["cyan"]} showLabel label="All providers" />,
    );
    expect(container.querySelector("text.fill-foreground")?.textContent).toBe("All providers");
  });

  it("renders no center label when data is empty", () => {
    const { container } = render(
      <DonutChart data={[]} index="provider" category="spend" colors={["cyan"]} showLabel />,
    );
    expect(container.querySelector("text.fill-foreground")).toBeNull();
  });

  const firstPathPoint = (container: HTMLElement) => {
    const d = container.querySelector(".recharts-pie-sector path")?.getAttribute("d") ?? "";
    const match = d.match(/M\s*([\d.-]+)\s*,\s*([\d.-]+)/);
    expect(match).not.toBeNull();
    return { x: Number(match![1]), y: Number(match![2]) };
  };

  it("starts the first sector at 3 o'clock by default and at 12 o'clock with tremor's 90/-270 angles", () => {
    const { container: byDefault } = render(
      <DonutChart data={data} index="provider" category="spend" colors={["cyan"]} />,
    );
    const { container: clockwiseFromTop } = render(
      <DonutChart data={data} index="provider" category="spend" colors={["cyan"]} startAngle={90} endAngle={-270} />,
    );

    const defaultStart = firstPathPoint(byDefault);
    const angledStart = firstPathPoint(clockwiseFromTop);
    expect(defaultStart.x).toBeGreaterThan(400);
    expect(Math.abs(defaultStart.y - 200)).toBeLessThan(1);
    expect(Math.abs(angledStart.x - 400)).toBeLessThan(1);
    expect(angledStart.y).toBeLessThan(200);
  });
});
