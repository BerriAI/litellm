import { render } from "@testing-library/react";
import React from "react";
import { describe, expect, it } from "vitest";
import { AreaChart } from "./area_chart";

const data = [
  { date: "2026-03-01", tokens: 100, requests: 10 },
  { date: "2026-03-02", tokens: 150, requests: 12 },
];

describe("AreaChart", () => {
  it("renders one area per category with the mapped stroke colors", () => {
    const { container } = render(
      <AreaChart data={data} index="date" categories={["tokens", "requests"]} colors={["blue", "cyan"]} />,
    );

    const curves = Array.from(container.querySelectorAll("path.recharts-area-curve"));
    expect(curves).toHaveLength(2);
    const strokes = new Set(curves.map((curve) => curve.getAttribute("stroke")));
    expect(strokes).toEqual(new Set(["var(--color-blue-500, #3b82f6)", "var(--color-cyan-500, #06b6d4)"]));
  });

  it("renders a fade-out gradient fill per category", () => {
    const { container } = render(
      <AreaChart data={data} index="date" categories={["tokens", "requests"]} colors={["blue", "cyan"]} />,
    );

    const gradients = container.querySelectorAll("defs linearGradient");
    expect(gradients).toHaveLength(2);
    const areas = Array.from(container.querySelectorAll("path.recharts-area-area"));
    expect(areas).toHaveLength(2);
    for (const area of areas) {
      expect(area.getAttribute("fill")).toMatch(/^url\(#fill-/);
    }
  });
});
