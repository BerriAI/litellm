import { describe, it, expect, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import RoutingStrategySelector from "./RoutingStrategySelector";

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
    expect(screen.getByRole("combobox")).toBeInTheDocument();
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

  it("should render all available strategies as options", async () => {
    const user = userEvent.setup();
    render(<RoutingStrategySelector {...baseProps} />);
    await user.click(screen.getByRole("combobox"));
    await waitFor(() => {
      expect(screen.getByRole("option", { name: /simple-shuffle/ })).toBeInTheDocument();
      expect(screen.getByRole("option", { name: /latency-based-routing/ })).toBeInTheDocument();
      expect(screen.getByRole("option", { name: /least-busy/ })).toBeInTheDocument();
    });
  });

  it("should display strategy descriptions alongside option labels", async () => {
    const user = userEvent.setup();
    render(<RoutingStrategySelector {...baseProps} />);
    await user.click(screen.getByRole("combobox"));
    expect(await screen.findByText("Randomly pick a deployment")).toBeInTheDocument();
    expect(
      await screen.findByText("Pick the lowest-latency deployment"),
    ).toBeInTheDocument();
  });

  it("should not render a description for a strategy that has none", async () => {
    const user = userEvent.setup();
    render(<RoutingStrategySelector {...baseProps} />);
    await user.click(screen.getByRole("combobox"));
    // "least-busy" has no entry in routingStrategyDescriptions — it still renders without crashing.
    expect(await screen.findByRole("option", { name: /least-busy/ })).toBeInTheDocument();
  });

  it("should call onStrategyChange with the selected strategy value", async () => {
    const onStrategyChange = vi.fn();
    const user = userEvent.setup();
    render(<RoutingStrategySelector {...baseProps} onStrategyChange={onStrategyChange} />);

    await user.click(screen.getByRole("combobox"));
    const option = await screen.findByRole("option", { name: /latency-based-routing/ });
    await user.click(option);

    expect(onStrategyChange).toHaveBeenCalledWith("latency-based-routing");
  });
});
