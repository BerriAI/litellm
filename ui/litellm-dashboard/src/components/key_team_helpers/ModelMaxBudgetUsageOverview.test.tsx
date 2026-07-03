import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ModelMaxBudgetUsageOverview } from "./ModelMaxBudgetUsageOverview";

describe("ModelMaxBudgetUsageOverview", () => {
  it("renders usage with percent and scope", () => {
    render(
      <ModelMaxBudgetUsageOverview
        usage={{
          "claude-sonnet-4-6": {
            current_spend: 15,
            budget_limit: 20,
            time_period: "1d",
            scope: "team",
            percent_used: 75,
          },
        }}
      />,
    );

    expect(screen.getByText("claude-sonnet-4-6")).toBeInTheDocument();
    expect(screen.getByText(/\$15\.00 \/ \$20\.00 \/ day/)).toBeInTheDocument();
    expect(screen.getByText("75%")).toBeInTheDocument();
    expect(screen.getByText("Shared across your keys on this team")).toBeInTheDocument();
  });

  it("shows empty state when no usage", () => {
    render(<ModelMaxBudgetUsageOverview usage={{}} />);
    expect(screen.getByText("No per-model budget usage to show")).toBeInTheDocument();
  });
});
