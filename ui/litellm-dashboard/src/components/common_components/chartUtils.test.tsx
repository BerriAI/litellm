import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { CustomLegend, CustomTooltip } from "./chartUtils";
import type { CustomTooltipProps } from "@tremor/react";
import { SpendMetrics } from "../UsagePage/types";

describe("CustomTooltip", () => {
  const mockPayload = [
    {
      dataKey: "metrics.total_tokens",
      value: 1000,
      color: "blue",
      payload: {
        date: "2024-01-15",
        metrics: {
          total_tokens: 1000,
          prompt_tokens: 600,
          completion_tokens: 400,
          spend: 0.05,
          api_requests: 10,
          successful_requests: 9,
          failed_requests: 1,
          cache_read_input_tokens: 0,
          cache_creation_input_tokens: 0,
        } as SpendMetrics,
      },
    },
  ];

  it("should render", () => {
    const props: CustomTooltipProps = {
      active: true,
      payload: mockPayload,
      label: "2024-01-15",
    };
    render(<CustomTooltip {...props} />);
    expect(screen.getByText("2024-01-15")).toBeInTheDocument();
  });

  it("should return null when not active", () => {
    const props: CustomTooltipProps = {
      active: false,
      payload: mockPayload,
      label: "2024-01-15",
    };
    const { container } = render(<CustomTooltip {...props} />);
    expect(container.firstChild).toBeNull();
  });

  it("should return null when payload is empty", () => {
    const props: CustomTooltipProps = {
      active: true,
      payload: [],
      label: "2024-01-15",
    };
    const { container } = render(<CustomTooltip {...props} />);
    expect(container.firstChild).toBeNull();
  });

  it("should display formatted category names", () => {
    const props: CustomTooltipProps = {
      active: true,
      payload: mockPayload,
      label: "2024-01-15",
    };
    render(<CustomTooltip {...props} />);
    expect(screen.getByText("Total Tokens")).toBeInTheDocument();
  });

  it("should format category names by removing metrics prefix and replacing underscores", () => {
    const payloadWithUnderscores = [
      {
        dataKey: "metrics.prompt_tokens",
        value: 600,
        color: "green",
        payload: {
          date: "2024-01-15",
          metrics: {
            prompt_tokens: 600,
            total_tokens: 1000,
            completion_tokens: 400,
            spend: 0.05,
            api_requests: 10,
            successful_requests: 9,
            failed_requests: 1,
            cache_read_input_tokens: 0,
            cache_creation_input_tokens: 0,
          } as SpendMetrics,
        },
      },
    ];
    const props: CustomTooltipProps = {
      active: true,
      payload: payloadWithUnderscores,
      label: "2024-01-15",
    };
    render(<CustomTooltip {...props} />);
    expect(screen.getByText("Prompt Tokens")).toBeInTheDocument();
  });

  it("should format spend values with dollar sign and two decimal places", () => {
    const spendPayload = [
      {
        dataKey: "metrics.spend",
        value: 1234.567,
        color: "red",
        payload: {
          date: "2024-01-15",
          metrics: {
            spend: 1234.567,
            total_tokens: 1000,
            prompt_tokens: 600,
            completion_tokens: 400,
            api_requests: 10,
            successful_requests: 9,
            failed_requests: 1,
            cache_read_input_tokens: 0,
            cache_creation_input_tokens: 0,
          } as SpendMetrics,
        },
      },
    ];
    const props: CustomTooltipProps = {
      active: true,
      payload: spendPayload,
      label: "2024-01-15",
    };
    render(<CustomTooltip {...props} />);
    expect(screen.getByText("$1,234.57")).toBeInTheDocument();
  });

  it("should format non-spend numeric values with locale string", () => {
    const props: CustomTooltipProps = {
      active: true,
      payload: mockPayload,
      label: "2024-01-15",
    };
    render(<CustomTooltip {...props} />);
    expect(screen.getByText("1,000")).toBeInTheDocument();
  });

  it("should display N/A when value is undefined", () => {
    const payloadWithUndefined = [
      {
        dataKey: "metrics.nonexistent",
        value: undefined,
        color: "blue",
        payload: {
          date: "2024-01-15",
          metrics: {
            total_tokens: 1000,
            prompt_tokens: 600,
            completion_tokens: 400,
            spend: 0.05,
            api_requests: 10,
            successful_requests: 9,
            failed_requests: 1,
            cache_read_input_tokens: 0,
            cache_creation_input_tokens: 0,
          } as SpendMetrics,
        },
      },
    ];
    const props: CustomTooltipProps = {
      active: true,
      payload: payloadWithUndefined,
      label: "2024-01-15",
    };
    render(<CustomTooltip {...props} />);
    expect(screen.getByText("N/A")).toBeInTheDocument();
  });

  it("should handle multiple payload items", () => {
    const multiplePayload = [
      {
        dataKey: "metrics.total_tokens",
        value: 1000,
        color: "blue",
        payload: {
          date: "2024-01-15",
          metrics: {
            total_tokens: 1000,
            prompt_tokens: 600,
            completion_tokens: 400,
            spend: 0.05,
            api_requests: 10,
            successful_requests: 9,
            failed_requests: 1,
            cache_read_input_tokens: 0,
            cache_creation_input_tokens: 0,
          } as SpendMetrics,
        },
      },
      {
        dataKey: "metrics.spend",
        value: 0.05,
        color: "green",
        payload: {
          date: "2024-01-15",
          metrics: {
            total_tokens: 1000,
            prompt_tokens: 600,
            completion_tokens: 400,
            spend: 0.05,
            api_requests: 10,
            successful_requests: 9,
            failed_requests: 1,
            cache_read_input_tokens: 0,
            cache_creation_input_tokens: 0,
          } as SpendMetrics,
        },
      },
    ];
    const props: CustomTooltipProps = {
      active: true,
      payload: multiplePayload,
      label: "2024-01-15",
    };
    render(<CustomTooltip {...props} />);
    expect(screen.getByText("Total Tokens")).toBeInTheDocument();
    expect(screen.getByText("Spend")).toBeInTheDocument();
  });

  it("should convert color names to hex values", () => {
    const props: CustomTooltipProps = {
      active: true,
      payload: mockPayload,
      label: "2024-01-15",
    };
    const { container } = render(<CustomTooltip {...props} />);
    const colorIndicator = container.querySelector('span[style*="background-color"]');
    expect(colorIndicator).toHaveStyle({ backgroundColor: "#3b82f6" });
  });

  it("should use hex color directly when color is not a known color name", () => {
    const payloadWithHexColor = [
      {
        dataKey: "metrics.total_tokens",
        value: 1000,
        color: "#ff0000",
        payload: {
          date: "2024-01-15",
          metrics: {
            total_tokens: 1000,
            prompt_tokens: 600,
            completion_tokens: 400,
            spend: 0.05,
            api_requests: 10,
            successful_requests: 9,
            failed_requests: 1,
            cache_read_input_tokens: 0,
            cache_creation_input_tokens: 0,
          } as SpendMetrics,
        },
      },
    ];
    const props: CustomTooltipProps = {
      active: true,
      payload: payloadWithHexColor,
      label: "2024-01-15",
    };
    const { container } = render(<CustomTooltip {...props} />);
    const colorIndicator = container.querySelector('span[style*="background-color"]');
    expect(colorIndicator).toHaveStyle({ backgroundColor: "#ff0000" });
  });

  it("should skip items without dataKey", () => {
    const payloadWithoutDataKey = [
      {
        dataKey: undefined,
        value: 1000,
        color: "blue",
        payload: {
          date: "2024-01-15",
          metrics: {
            total_tokens: 1000,
            prompt_tokens: 600,
            completion_tokens: 400,
            spend: 0.05,
            api_requests: 10,
            successful_requests: 9,
            failed_requests: 1,
            cache_read_input_tokens: 0,
            cache_creation_input_tokens: 0,
          } as SpendMetrics,
        },
      },
    ];
    const props: CustomTooltipProps = {
      active: true,
      payload: payloadWithoutDataKey as any,
      label: "2024-01-15",
    };
    render(<CustomTooltip {...props} />);
    expect(screen.queryByText("Total Tokens")).not.toBeInTheDocument();
  });

  it("should skip items without payload", () => {
    const payloadWithoutPayload = [
      {
        dataKey: "metrics.total_tokens",
        value: 1000,
        color: "blue",
        payload: undefined,
      },
    ];
    const props: CustomTooltipProps = {
      active: true,
      payload: payloadWithoutPayload as any,
      label: "2024-01-15",
    };
    render(<CustomTooltip {...props} />);
    expect(screen.queryByText("Total Tokens")).not.toBeInTheDocument();
  });
});

describe("CustomLegend", () => {
  it("should render", () => {
    render(<CustomLegend categories={["metrics.total_tokens"]} colors={["blue"]} />);
    expect(screen.getByText("Total Tokens")).toBeInTheDocument();
  });

  it("should display multiple categories", () => {
    render(
      <CustomLegend
        categories={["metrics.total_tokens", "metrics.spend", "metrics.prompt_tokens"]}
        colors={["blue", "green", "red"]}
      />,
    );
    expect(screen.getByText("Total Tokens")).toBeInTheDocument();
    expect(screen.getByText("Spend")).toBeInTheDocument();
    expect(screen.getByText("Prompt Tokens")).toBeInTheDocument();
  });

  it("should format category names by removing metrics prefix and replacing underscores", () => {
    render(<CustomLegend categories={["metrics.api_requests"]} colors={["blue"]} />);
    expect(screen.getByText("Api Requests")).toBeInTheDocument();
  });

  it("should capitalize first letter of each word", () => {
    render(<CustomLegend categories={["metrics.successful_requests"]} colors={["green"]} />);
    expect(screen.getByText("Successful Requests")).toBeInTheDocument();
  });

  it("should convert color names to hex values", () => {
    const { container } = render(<CustomLegend categories={["metrics.total_tokens"]} colors={["cyan"]} />);
    const colorIndicator = container.querySelector('span[style*="background-color"]');
    expect(colorIndicator).toHaveStyle({ backgroundColor: "#06b6d4" });
  });

  it("should use hex color directly when color is not a known color name", () => {
    const { container } = render(<CustomLegend categories={["metrics.total_tokens"]} colors={["#ff00ff"]} />);
    const colorIndicator = container.querySelector('span[style*="background-color"]');
    expect(colorIndicator).toHaveStyle({ backgroundColor: "#ff00ff" });
  });

  it("should handle all supported color names", () => {
    const colors = ["blue", "cyan", "indigo", "green", "red", "purple", "emerald"];
    const categories = colors.map((_, idx) => `metrics.category_${idx}`);
    render(<CustomLegend categories={categories} colors={colors} />);
    expect(screen.getByText("Category 0")).toBeInTheDocument();
  });

  it("should match categories and colors by index", () => {
    render(
      <CustomLegend
        categories={["metrics.first", "metrics.second", "metrics.third"]}
        colors={["blue", "green", "red"]}
      />,
    );
    const { container } = render(
      <CustomLegend
        categories={["metrics.first", "metrics.second", "metrics.third"]}
        colors={["blue", "green", "red"]}
      />,
    );
    const colorIndicators = container.querySelectorAll('span[style*="background-color"]');
    expect(colorIndicators[0]).toHaveStyle({ backgroundColor: "#3b82f6" });
    expect(colorIndicators[1]).toHaveStyle({ backgroundColor: "#22c55e" });
    expect(colorIndicators[2]).toHaveStyle({ backgroundColor: "#ef4444" });
  });
});
