import React from "react";
import { describe, it, expect } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "@/../tests/test-utils";
import AgentCostView from "./agent_cost_view";
import type { Agent } from "@/components/agents/types";

const makeAgent = (litellmParams: Agent["litellm_params"]): Agent => ({
  agent_id: "agent-1",
  agent_name: "Test Agent",
  litellm_params: litellmParams,
});

describe("AgentCostView", () => {
  it("renders nothing when the agent has no cost configuration at all", () => {
    const { container } = renderWithProviders(<AgentCostView agent={makeAgent({ model: "gpt-4" })} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders every configured cost with a dollar-prefixed value", () => {
    const fullyPricedParams = {
      model: "gpt-4",
      cost_per_query: 0.05,
      input_cost_per_token: 0.000012,
      output_cost_per_token: 0.000034,
    };
    renderWithProviders(<AgentCostView agent={makeAgent(fullyPricedParams)} />);

    expect(screen.getByText("Cost Configuration")).toBeInTheDocument();
    expect(screen.getByText("Cost Per Query")).toBeInTheDocument();
    expect(screen.getByText("$0.05")).toBeInTheDocument();
    expect(screen.getByText("Input Cost Per Token")).toBeInTheDocument();
    expect(screen.getByText("$0.000012")).toBeInTheDocument();
    expect(screen.getByText("Output Cost Per Token")).toBeInTheDocument();
    expect(screen.getByText("$0.000034")).toBeInTheDocument();
  });

  it("omits the rows whose cost is not configured", () => {
    renderWithProviders(<AgentCostView agent={makeAgent({ model: "gpt-4", cost_per_query: 0.25 })} />);

    expect(screen.getByText("Cost Per Query")).toBeInTheDocument();
    expect(screen.getByText("$0.25")).toBeInTheDocument();
    expect(screen.queryByText("Input Cost Per Token")).not.toBeInTheDocument();
    expect(screen.queryByText("Output Cost Per Token")).not.toBeInTheDocument();
  });

  it("still renders a zero cost rather than treating it as unset", () => {
    renderWithProviders(<AgentCostView agent={makeAgent({ model: "gpt-4", cost_per_query: 0 })} />);

    expect(screen.getByText("Cost Configuration")).toBeInTheDocument();
    expect(screen.getByText("Cost Per Query")).toBeInTheDocument();
    expect(screen.getByText("$0")).toBeInTheDocument();
  });
});
