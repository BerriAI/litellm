import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import RouterSettingsForm from "./RouterSettingsForm";
import type { RouterSettingsFormValue } from "./RouterSettingsForm";

// Use the same antd mock as RoutingStrategySelector to keep things consistent
vi.mock("antd", () => ({
  Select: Object.assign(
    ({ value, onChange, children }: any) => (
      <select
        data-testid="strategy-select"
        value={value ?? ""}
        onChange={(e) => onChange(e.target.value)}
      >
        {children}
      </select>
    ),
    {
      Option: ({ value, children }: any) => (
        <option value={value}>{children}</option>
      ),
    }
  ),
}));

vi.mock("@tremor/react", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@tremor/react")>();
  return {
    ...actual,
    Switch: ({ checked, onChange }: any) => (
      <input
        type="checkbox"
        role="switch"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
    ),
  };
});

const defaultValue: RouterSettingsFormValue = {
  routerSettings: {},
  selectedStrategy: null,
  enableTagFiltering: false,
};

const baseProps = {
  value: defaultValue,
  onChange: vi.fn(),
  routerFieldsMetadata: {},
  availableRoutingStrategies: [],
  routingStrategyDescriptions: {},
};

describe("RouterSettingsForm", () => {
  it("should render", () => {
    render(<RouterSettingsForm {...baseProps} />);
    expect(screen.getByText("Routing Settings")).toBeInTheDocument();
  });

  it("should not show the strategy selector when no strategies are provided", () => {
    render(<RouterSettingsForm {...baseProps} />);
    expect(screen.queryByTestId("strategy-select")).not.toBeInTheDocument();
  });

  it("should show the strategy selector when strategies are available", () => {
    const props = {
      ...baseProps,
      availableRoutingStrategies: ["simple-shuffle", "latency-based-routing"],
    };
    render(<RouterSettingsForm {...props} />);
    expect(screen.getByTestId("strategy-select")).toBeInTheDocument();
  });

  it("should not render LatencyBasedConfiguration for non-latency strategies", () => {
    const props = {
      ...baseProps,
      value: { ...defaultValue, selectedStrategy: "simple-shuffle" },
      availableRoutingStrategies: ["simple-shuffle"],
    };
    render(<RouterSettingsForm {...props} />);
    expect(screen.queryByText("Latency-Based Configuration")).not.toBeInTheDocument();
  });

  it("should render LatencyBasedConfiguration when strategy is latency-based-routing", () => {
    const props = {
      ...baseProps,
      value: {
        ...defaultValue,
        selectedStrategy: "latency-based-routing",
        routerSettings: { routing_strategy_args: { ttl: 3600, lowest_latency_buffer: 0 } },
      },
      availableRoutingStrategies: ["latency-based-routing"],
    };
    render(<RouterSettingsForm {...props} />);
    expect(screen.getByText("Latency-Based Configuration")).toBeInTheDocument();
  });

  it("should call onChange with the updated strategy when the selector changes", async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    const props = {
      ...baseProps,
      onChange,
      availableRoutingStrategies: ["simple-shuffle", "latency-based-routing"],
    };
    render(<RouterSettingsForm {...props} />);

    await user.selectOptions(screen.getByTestId("strategy-select"), "latency-based-routing");

    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({ selectedStrategy: "latency-based-routing" })
    );
  });

  it("should call onChange with the updated enableTagFiltering when the toggle changes", async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(<RouterSettingsForm {...baseProps} onChange={onChange} />);

    await user.click(screen.getByRole("switch"));

    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({ enableTagFiltering: true })
    );
  });

  it("should show the Reliability & Retries section", () => {
    render(<RouterSettingsForm {...baseProps} />);
    expect(screen.getByText("Reliability & Retries")).toBeInTheDocument();
  });
});
