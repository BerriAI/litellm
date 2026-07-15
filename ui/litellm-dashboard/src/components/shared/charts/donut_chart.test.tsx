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
});
