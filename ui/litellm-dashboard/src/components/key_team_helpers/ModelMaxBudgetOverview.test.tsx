import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { KeyResponse } from "./key_list";
import {
  collectModelMaxBudgetOverrides,
  ModelMaxBudgetOverview,
  parseModelMaxBudgetValue,
} from "./ModelMaxBudgetOverview";

const createMockKey = (overrides: Partial<KeyResponse> = {}): KeyResponse =>
  ({
    token: "sk-test",
    token_id: "key-1",
    key_alias: "test-key",
    key_name: "sk-...test",
    spend: 0,
    max_budget: 100,
    ...overrides,
  }) as KeyResponse;

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

  it("parses legacy numeric key budgets from KeyResponse shape", () => {
    const rows = collectModelMaxBudgetOverrides(
      [],
      [
        createMockKey({
          token_id: "key-1",
          key_alias: "legacy-key",
          model_max_budget: { "claude-opus-4-8": 15 },
        }),
      ],
    );

    expect(rows).toHaveLength(1);
    expect(rows[0]?.entries[0]?.model).toBe("claude-opus-4-8");
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
        createMockKey({
          token_id: "key-1",
          key_alias: "dev-key",
          model_max_budget: {
            "claude-sonnet-4-6": { budget_limit: 10, time_period: "1d" },
          } as unknown as KeyResponse["model_max_budget"],
        }),
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
          createMockKey({
            token_id: "key-1",
            key_alias: "local-dev",
            model_max_budget: {
              "claude-sonnet-4-6": { budget_limit: 5, time_period: "1d" },
            } as unknown as KeyResponse["model_max_budget"],
          }),
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
