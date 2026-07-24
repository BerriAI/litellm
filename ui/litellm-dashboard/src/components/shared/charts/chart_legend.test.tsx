import { render, screen } from "@testing-library/react";
import React from "react";
import { describe, expect, it } from "vitest";
import { CustomLegend } from "./chart_legend";

describe("CustomLegend", () => {
  it("renders title-cased category names without the metrics prefix", () => {
    render(<CustomLegend categories={["metrics.total_tokens", "metrics.spend"]} colors={["blue", "green"]} />);

    expect(screen.getByText("Total Tokens")).toBeInTheDocument();
    expect(screen.getByText("Spend")).toBeInTheDocument();
  });

  it("matches colors to categories by index with theme-var values", () => {
    const { container } = render(
      <CustomLegend categories={["metrics.first", "metrics.second"]} colors={["blue", "green"]} />,
    );

    const dots = Array.from(container.querySelectorAll('span[style*="background-color"]'));
    expect(dots[0]?.getAttribute("style")).toContain("--color-blue-500");
    expect(dots[1]?.getAttribute("style")).toContain("--color-green-500");
  });

  it("cycles colors when there are more categories than colors", () => {
    const { container } = render(
      <CustomLegend categories={["metrics.a", "metrics.b", "metrics.c"]} colors={["blue", "green"]} />,
    );

    const dots = Array.from(container.querySelectorAll('span[style*="background-color"]'));
    expect(dots[2]?.getAttribute("style")).toContain("--color-blue-500");
  });
});
