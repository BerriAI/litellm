import { render, screen } from "@testing-library/react";
import React from "react";
import { describe, expect, it } from "vitest";
import { CustomTooltip, ValueTooltip, type ChartTooltipProps } from "./chart_tooltip";

const metricsPayload = (
  dataKey: string,
  value: number,
  color = "#3b82f6",
): NonNullable<ChartTooltipProps["payload"]>[number] =>
  ({
    dataKey,
    value,
    color,
    payload: {
      date: "2026-01-15",
      metrics: {
        total_tokens: 1000,
        prompt_tokens: 600,
        completion_tokens: 400,
        spend: 1234.567,
        api_requests: 10,
      },
    },
  }) as NonNullable<ChartTooltipProps["payload"]>[number];

describe("CustomTooltip", () => {
  it("returns null when not active or payload is empty", () => {
    const inactive = render(
      <CustomTooltip active={false} payload={[metricsPayload("metrics.total_tokens", 1000)]} label="2026-01-15" />,
    );
    expect(inactive.container.firstChild).toBeNull();

    const empty = render(<CustomTooltip active={true} payload={[]} label="2026-01-15" />);
    expect(empty.container.firstChild).toBeNull();
  });

  it("renders the label and title-cased category names without the metrics prefix", () => {
    render(<CustomTooltip active={true} payload={[metricsPayload("metrics.total_tokens", 1000)]} label="2026-01-15" />);

    expect(screen.getByText("2026-01-15")).toBeInTheDocument();
    expect(screen.getByText("Total Tokens")).toBeInTheDocument();
    expect(screen.getByText("1,000")).toBeInTheDocument();
  });

  it("formats spend values as dollars with two decimals", () => {
    render(<CustomTooltip active={true} payload={[metricsPayload("metrics.spend", 1234.567)]} label="2026-01-15" />);

    expect(screen.getByText("$1,234.57")).toBeInTheDocument();
  });

  it("shows N/A for metrics missing from the row payload", () => {
    render(<CustomTooltip active={true} payload={[metricsPayload("metrics.nonexistent", 1000)]} label="2026-01-15" />);

    expect(screen.getByText("N/A")).toBeInTheDocument();
  });

  it("uses the series color for the indicator dot", () => {
    const { container } = render(
      <CustomTooltip
        active={true}
        payload={[metricsPayload("metrics.total_tokens", 1000, "var(--color-blue-500, #3b82f6)")]}
        label="2026-01-15"
      />,
    );

    const dot = container.querySelector('span[style*="background-color"]');
    expect(dot?.getAttribute("style")).toContain("--color-blue-500");
  });
});

describe("ValueTooltip", () => {
  const payload = [
    {
      dataKey: "passed",
      name: "passed",
      value: 1000,
      color: "#22c55e",
      payload: { date: "2026-01-15", passed: 1000 },
    } as NonNullable<ChartTooltipProps["payload"]>[number],
  ];

  it("returns null when not active", () => {
    const { container } = render(<ValueTooltip active={false} payload={payload} label="2026-01-15" />);
    expect(container.firstChild).toBeNull();
  });

  it("renders label, series name, and locale-formatted value by default", () => {
    render(<ValueTooltip active={true} payload={payload} label="2026-01-15" />);

    expect(screen.getByText("2026-01-15")).toBeInTheDocument();
    expect(screen.getByText("passed")).toBeInTheDocument();
    expect(screen.getByText("1,000")).toBeInTheDocument();
  });

  it("applies the valueFormatter to values", () => {
    render(<ValueTooltip active={true} payload={payload} label="2026-01-15" valueFormatter={(v) => `$${v}`} />);

    expect(screen.getByText("$1000")).toBeInTheDocument();
  });
});
