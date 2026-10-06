import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import SubscriptionCostCard from "./SubscriptionCostCard";
import type { SubscriptionAccountUsage, SubscriptionFee } from "./types";

vi.mock("@/components/shared/chart_loader", () => ({
  ChartLoader: () => <div data-testid="chart-loader">Loading chart data...</div>,
}));

vi.mock("@/components/molecules/models/ProviderLogo", () => ({
  ProviderLogo: ({ provider }: { provider: string }) => <div data-testid={`provider-logo-${provider}`} />,
}));

const fee = (overrides: Partial<SubscriptionFee> = {}): SubscriptionFee => ({
  subscription_account_id: "fee-1",
  custom_llm_provider: "chatgpt",
  account_id: "acct_123",
  label: null,
  monthly_fee: 25,
  currency: "USD",
  billing_period_start: "2026-10-01",
  created_at: "2026-10-06T00:00:00Z",
  updated_at: "2026-10-06T00:00:00Z",
  created_by: "admin",
  updated_by: "admin",
  ...overrides,
});

const account = (overrides: Partial<SubscriptionAccountUsage> = {}): SubscriptionAccountUsage => ({
  custom_llm_provider: "chatgpt",
  account_id: "acct_123",
  deployments: ["gpt-5.6-terra", "gpt-5.6-sol"],
  fee: null,
  billing_periods: [],
  fixed_cost: null,
  ...overrides,
});

const handlers = () => ({ onSetFee: vi.fn(), onEditFee: vi.fn(), onRemoveFee: vi.fn() });

const renderCard = (accounts: SubscriptionAccountUsage[], props: { canEdit?: boolean; meteredSpend?: number } = {}) => {
  const actions = handlers();
  const view = render(
    <SubscriptionCostCard
      loading={false}
      failed={false}
      isDateChanging={false}
      usage={{ subscription_providers: ["chatgpt"], accounts }}
      meteredSpend={props.meteredSpend ?? 0}
      canEdit={props.canEdit ?? true}
      {...actions}
    />,
  );
  return { ...view, ...actions };
};

describe("SubscriptionCostCard", () => {
  it("renders nothing when there are no subscription accounts", () => {
    const { container } = render(
      <SubscriptionCostCard
        loading={false}
        failed={false}
        isDateChanging={false}
        usage={{ subscription_providers: [], accounts: [] }}
        meteredSpend={0}
        canEdit
        {...handlers()}
      />,
    );
    expect(container).toBeEmptyDOMElement();
    expect(screen.queryByText("Subscription fixed costs")).not.toBeInTheDocument();
  });

  it("calls out an untracked fixed cost and offers to set the fee only to an editor", () => {
    const { onSetFee, unmount } = renderCard([account()]);
    expect(screen.getByText("Fixed cost not tracked")).toBeInTheDocument();
    expect(screen.getByText("gpt-5.6-terra, gpt-5.6-sol")).toBeInTheDocument();
    expect(screen.getByText("None in range")).toBeInTheDocument();
    expect(screen.getByTestId("fixed-cost-total")).toHaveTextContent("Fixed cost in range: none");
    fireEvent.click(screen.getByRole("button", { name: "Set monthly fee" }));
    expect(onSetFee).toHaveBeenCalledWith(account());
    unmount();

    renderCard([account()], { canEdit: false });
    expect(screen.getByText("Fixed cost not tracked")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Set monthly fee" })).not.toBeInTheDocument();
  });

  it("shows a configured fee once for two deployments sharing the account and the fixed cost in range", () => {
    const covered = account({
      fee: fee({ label: "Team workspace" }),
      billing_periods: [{ start: "2026-10-01", end: "2026-10-31" }],
      fixed_cost: 25,
    });
    const { onEditFee, onRemoveFee } = renderCard([covered], { meteredSpend: 0.0838 });

    expect(screen.getAllByText("25.00 USD")).toHaveLength(2);
    expect(screen.getByText("Team workspace")).toBeInTheDocument();
    expect(screen.getByText("Oct 1 - Oct 31, 2026")).toBeInTheDocument();
    expect(screen.getByTestId("fixed-cost-total")).toHaveTextContent("Fixed cost in range: 25.00 USD");
    expect(screen.getByTestId("metered-spend")).toHaveTextContent("Metered spend: $0.08");
    expect(screen.queryByText(/25\.08/)).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Edit" }));
    expect(onEditFee).toHaveBeenCalledWith(covered);
    fireEvent.click(screen.getByRole("button", { name: "Remove" }));
    expect(onRemoveFee).toHaveBeenCalledWith(covered);
  });

  it("sums fixed costs per currency and keeps a second currency separate", () => {
    const october = { start: "2026-10-01", end: "2026-10-31" };
    const twoPeriods = {
      fee: fee(),
      billing_periods: [{ start: "2026-09-01", end: "2026-09-30" }, october],
      fixed_cost: 50,
    };
    const secondUsd = { subscription_account_id: "fee-2", account_id: "acct_456", monthly_fee: 10 };
    const noDeployments = {
      account_id: "acct_456",
      deployments: [],
      fee: fee(secondUsd),
      billing_periods: [october],
      fixed_cost: 10,
    };
    const euro = { subscription_account_id: "fee-3", account_id: "acct_789", monthly_fee: 20, currency: "EUR" };
    const inEuros = { account_id: "acct_789", fee: fee(euro), billing_periods: [october], fixed_cost: 20 };
    renderCard([account(twoPeriods), account(noDeployments), account(inEuros)]);

    const totals = screen.getAllByTestId("fixed-cost-total").map((node) => node.textContent);
    expect(totals).toEqual(["Fixed cost in range: 60.00 USD", "Fixed cost in range: 20.00 EUR"]);
    expect(screen.getByText("None")).toBeInTheDocument();
    expect(screen.getByText("Sep 1 - Sep 30, 2026")).toBeInTheDocument();
  });

  it("explains an unresolved account and offers no action for it", () => {
    renderCard([account({ account_id: null })]);
    expect(screen.getByText("Account not signed in on the proxy")).toBeInTheDocument();
    expect(screen.getByText("Fixed cost not tracked")).toBeInTheDocument();
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });

  it("shows the loader instead of the table while a range is loading", () => {
    render(
      <SubscriptionCostCard
        loading
        failed={false}
        isDateChanging
        usage={{ subscription_providers: ["chatgpt"], accounts: [account({ fee: fee(), fixed_cost: 25 })] }}
        meteredSpend={0}
        canEdit
        {...handlers()}
      />,
    );
    expect(screen.getByTestId("chart-loader")).toBeInTheDocument();
    expect(screen.queryByText("25.00 USD")).not.toBeInTheDocument();
  });

  it("says so instead of disappearing when the usage request fails", () => {
    render(
      <SubscriptionCostCard
        loading={false}
        failed
        isDateChanging={false}
        usage={null}
        meteredSpend={0}
        canEdit
        {...handlers()}
      />,
    );
    expect(screen.getByText("Could not load subscription accounts for this range.")).toBeInTheDocument();
    expect(screen.queryByTestId("chart-loader")).not.toBeInTheDocument();
  });
});
