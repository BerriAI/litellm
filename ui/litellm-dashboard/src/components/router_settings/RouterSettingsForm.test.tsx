import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import RouterSettingsForm from "./RouterSettingsForm";
import type { RouterSettingsFormValue } from "./RouterSettingsForm";

// The strategy selector is a shadcn Select (role="combobox"). Additional
// antd components (Switch, Button, …) render from the real antd package.

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
    expect(screen.queryByRole("combobox")).not.toBeInTheDocument();
  });

  it("should show the strategy selector when strategies are available", () => {
    const props = {
      ...baseProps,
      availableRoutingStrategies: ["simple-shuffle", "latency-based-routing"],
    };
    render(<RouterSettingsForm {...props} />);
    expect(screen.getByRole("combobox")).toBeInTheDocument();
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

    await user.click(screen.getByRole("combobox"));
    const option = await screen.findByRole("option", { name: /latency-based-routing/ });
    await user.click(option);

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
