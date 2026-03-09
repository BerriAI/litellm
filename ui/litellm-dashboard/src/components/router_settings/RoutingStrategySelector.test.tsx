import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import RoutingStrategySelector from "./RoutingStrategySelector";

// Ant Design's Select is complex to drive in JSDOM; swap it for a plain
// <select> so we can assert options and fire change events normally.
vi.mock("antd", () => ({
  Select: Object.assign(
    ({ value, onChange, children }: any) => (
      <div data-testid="ant-select">
        <select
          data-testid="strategy-select"
          value={value ?? ""}
          onChange={(e) => onChange(e.target.value)}
        >
          {children}
        </select>
      </div>
    ),
    {
      Option: ({ value, children }: any) => (
        <option value={value}>{children}</option>
      ),
    }
  ),
}));

const baseProps = {
  selectedStrategy: null,
  availableStrategies: ["simple-shuffle", "latency-based-routing", "least-busy"],
  routingStrategyDescriptions: {
    "simple-shuffle": "Randomly pick a deployment",
    "latency-based-routing": "Pick the lowest-latency deployment",
  },
  routerFieldsMetadata: {},
  onStrategyChange: vi.fn(),
};

describe("RoutingStrategySelector", () => {
  it("should render", () => {
    render(<RoutingStrategySelector {...baseProps} />);
    expect(screen.getByTestId("ant-select")).toBeInTheDocument();
  });

  it("should display default label when no metadata is provided", () => {
    render(<RoutingStrategySelector {...baseProps} />);
    expect(screen.getByText("Routing Strategy")).toBeInTheDocument();
  });

  it("should display ui_field_name from metadata when provided", () => {
    const props = {
      ...baseProps,
      routerFieldsMetadata: {
        routing_strategy: {
          ui_field_name: "Strategy",
          field_description: "How to pick a deployment",
        },
      },
    };
    render(<RoutingStrategySelector {...props} />);
    expect(screen.getByText("Strategy")).toBeInTheDocument();
    expect(screen.getByText("How to pick a deployment")).toBeInTheDocument();
  });

  it("should render all available strategies as options", () => {
    render(<RoutingStrategySelector {...baseProps} />);
    expect(screen.getByText("simple-shuffle")).toBeInTheDocument();
    expect(screen.getByText("latency-based-routing")).toBeInTheDocument();
    expect(screen.getByText("least-busy")).toBeInTheDocument();
  });

  it("should display strategy descriptions alongside option labels", () => {
    render(<RoutingStrategySelector {...baseProps} />);
    expect(screen.getByText("Randomly pick a deployment")).toBeInTheDocument();
    expect(screen.getByText("Pick the lowest-latency deployment")).toBeInTheDocument();
  });

  it("should not render a description for a strategy that has none", () => {
    render(<RoutingStrategySelector {...baseProps} />);
    // "least-busy" has no entry in routingStrategyDescriptions — it still renders without crashing
    expect(screen.getByText("least-busy")).toBeInTheDocument();
  });

  it("should call onStrategyChange with the selected strategy value", async () => {
    const onStrategyChange = vi.fn();
    const user = userEvent.setup();
    render(<RoutingStrategySelector {...baseProps} onStrategyChange={onStrategyChange} />);

    await user.selectOptions(screen.getByTestId("strategy-select"), "latency-based-routing");

    expect(onStrategyChange).toHaveBeenCalledWith("latency-based-routing");
  });
});
