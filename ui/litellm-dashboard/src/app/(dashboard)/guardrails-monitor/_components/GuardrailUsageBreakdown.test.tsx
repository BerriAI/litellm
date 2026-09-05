import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { GuardrailUsageDetail } from "@/app/(dashboard)/hooks/guardrails/useGuardrailsUsage";
import { GuardrailUsageBreakdown } from "./GuardrailUsageBreakdown";

const detail: GuardrailUsageDetail = {
  guardrail_id: "bedrock-pii-mask",
  guardrail_name: "bedrock-pii-mask",
  type: "pii",
  provider: "Bedrock",
  requestsEvaluated: 5,
  failRate: 0,
  avgScore: null,
  avgLatency: 120,
  status: "healthy",
  trend: "stable",
  description: null,
  time_series: [],
  usage_units: { contentPolicyUnits: 1000, sensitiveInformationPolicyUnits: 300, someFutureCounter: 7 },
  usage_units_daily: [],
  usage_units_by_team: {
    "team-a": { contentPolicyUnits: 900, sensitiveInformationPolicyUnits: 300 },
    "": { contentPolicyUnits: 100, someFutureCounter: 7 },
  },
  usage_units_by_key: {
    "hash-1": { contentPolicyUnits: 1000, sensitiveInformationPolicyUnits: 300 },
    "hash-2": { someFutureCounter: 7 },
  },
  cost: 0.18,
  cost_by_unit: { contentPolicyUnits: 0.15, sensitiveInformationPolicyUnits: 0.03, someFutureCounter: null },
  cost_by_team: { "team-a": 0.165, "": 0.015 },
  cost_by_key: { "hash-1": 0.18, "hash-2": null },
  untracked_usage_units: { someFutureCounter: 7 },
  untracked_usage_units_by_team: { "team-a": {}, "": { someFutureCounter: 7 } },
  untracked_usage_units_by_key: { "hash-1": {}, "hash-2": { someFutureCounter: 7 } },
};

const rowNamed = (name: string) => screen.getByRole("row", { name: new RegExp(name) });

describe("GuardrailUsageBreakdown", () => {
  it("totals the units and the cost, and says how many units the cost leaves out", () => {
    render(<GuardrailUsageBreakdown detail={detail} />);

    const cost = screen.getByRole("group", { name: "Cost" });
    expect(cost).toHaveTextContent("$0.1800");
    expect(cost).toHaveTextContent("7 units unpriced");

    const units = screen.getByRole("group", { name: "Usage Units" });
    expect(units).toHaveTextContent("1,307");
    expect(units).toHaveTextContent("3 counters");
  });

  it("lists each counter with its units, cost and unpriced share", () => {
    render(<GuardrailUsageBreakdown detail={detail} />);

    const content = rowNamed("Content Policy");
    expect(within(content).getByText("1,000")).toBeInTheDocument();
    expect(within(content).getByText("$0.1500")).toBeInTheDocument();
    expect(within(content).getByText("—")).toBeInTheDocument();

    const future = rowNamed("Some Future Counter");
    expect(within(future).getByText("7", { selector: ".text-warning" })).toBeInTheDocument();
    expect(within(future).getByText("—")).toBeInTheDocument();
  });

  it("breaks units and cost down by team and by key, flagging the unpriced share of each row", () => {
    render(<GuardrailUsageBreakdown detail={detail} />);

    expect(screen.getByRole("heading", { name: "By team" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "By key" })).toBeInTheDocument();
    const teamA = rowNamed("team-a");
    expect(within(teamA).getByText("1,200")).toBeInTheDocument();
    expect(within(teamA).getByText("$0.1650")).toBeInTheDocument();
    expect(within(teamA).getByText("—")).toBeInTheDocument();
    expect(within(teamA).queryByText("7")).not.toBeInTheDocument();

    const noTeam = rowNamed("No team");
    expect(within(noTeam).getByText("107")).toBeInTheDocument();
    expect(within(noTeam).getByText("$0.0150")).toBeInTheDocument();
    expect(within(noTeam).getByText("7", { selector: ".text-warning" })).toBeInTheDocument();

    const unpricedKey = rowNamed("hash-2");
    expect(within(unpricedKey).getByText("—")).toBeInTheDocument();
    expect(within(unpricedKey).getByText("7", { selector: ".text-warning" })).toBeInTheDocument();
  });

  it("orders teams and keys by units, largest first", () => {
    render(<GuardrailUsageBreakdown detail={detail} />);

    const rows = screen.getAllByRole("row").map((row) => row.textContent ?? "");
    expect(rows.findIndex((text) => text.includes("team-a"))).toBeLessThan(
      rows.findIndex((text) => text.includes("No team")),
    );
    expect(rows.findIndex((text) => text.includes("hash-1"))).toBeLessThan(
      rows.findIndex((text) => text.includes("hash-2")),
    );
  });

  it("says so when the window has no billable units instead of rendering empty tables", () => {
    render(
      <GuardrailUsageBreakdown
        detail={{
          ...detail,
          usage_units: {},
          usage_units_by_team: {},
          usage_units_by_key: {},
          cost: null,
          cost_by_unit: {},
          cost_by_team: {},
          cost_by_key: {},
          untracked_usage_units: {},
          untracked_usage_units_by_team: {},
          untracked_usage_units_by_key: {},
        }}
      />,
    );

    expect(screen.getByText("No billable usage units were recorded in this period.")).toBeInTheDocument();
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
  });
});
