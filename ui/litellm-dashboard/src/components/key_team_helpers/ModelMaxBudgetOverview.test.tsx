import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { KeyResponse } from "./key_list";
import {
  collectModelMaxBudgetOverrides,
  ModelMaxBudgetOverview,
  parseModelMaxBudgetValue,
} from "./ModelMaxBudgetOverview";

describe("parseModelMaxBudgetValue", () => {
  it("parses budget config objects", () => {
    expect(
      parseModelMaxBudgetValue({
        "claude-sonnet-4-6": { budget_limit: 20, time_period: "1d" },
      }),
    ).toEqual({
      "claude-sonnet-4-6": { budget_limit: 20, time_period: "1d" },
    });
  });

  it("parses legacy numeric budgets", () => {
    expect(parseModelMaxBudgetValue({ "claude-sonnet-4-6": 20 })).toEqual({
      "claude-sonnet-4-6": { budget_limit: 20, time_period: "1d" },
    });
  });
});

describe("collectModelMaxBudgetOverrides", () => {
  it("collects member and key overrides separately", () => {
    const rows = collectModelMaxBudgetOverrides(
      [
        {
          user_id: "user-1",
          litellm_budget_table: {
            model_max_budget: {
              "claude-opus-4-8": { budget_limit: 50, time_period: "1d" },
            },
          },
        },
      ],
      [
        {
          token_id: "key-1",
          key_alias: "dev-key",
          model_max_budget: {
            "claude-sonnet-4-6": { budget_limit: 10, time_period: "1d" },
          },
        } as KeyResponse,
      ],
    );

    expect(rows).toHaveLength(2);
    expect(rows.find((row) => row.kind === "member")?.label).toBe("user-1");
    expect(rows.find((row) => row.kind === "key")?.label).toBe("dev-key");
  });
});

describe("ModelMaxBudgetOverview", () => {
  it("shows team defaults and override sections", () => {
    render(
      <ModelMaxBudgetOverview
        teamModelMaxBudget={{
          "claude-sonnet-4-6": { budget_limit: 20, time_period: "1d" },
        }}
        memberships={[
          {
            user_id: "user@bitovi.com",
            litellm_budget_table: {
              model_max_budget: {
                "claude-opus-4-8": { budget_limit: 50, time_period: "1d" },
              },
            },
          },
        ]}
        keys={[
          {
            token_id: "key-1",
            key_alias: "local-dev",
            model_max_budget: {
              "claude-sonnet-4-6": { budget_limit: 5, time_period: "1d" },
            },
          } as KeyResponse,
        ]}
      />,
    );

    expect(screen.getByText("Team defaults")).toBeInTheDocument();
    expect(screen.getByText(/claude-sonnet-4-6: \$20\.00 \/ day/)).toBeInTheDocument();
    expect(screen.getByText("Member overrides (1)")).toBeInTheDocument();
    expect(screen.getByText("user@bitovi.com")).toBeInTheDocument();
    expect(screen.getByText("Virtual key overrides (1)")).toBeInTheDocument();
    expect(screen.getByText("local-dev")).toBeInTheDocument();
    expect(screen.getByText(/Enforced per member and virtual key/)).toBeInTheDocument();
  });

  it("shows empty state when no budgets exist", () => {
    render(<ModelMaxBudgetOverview />);

    expect(screen.getByText("No per-model budgets configured")).toBeInTheDocument();
  });
});
