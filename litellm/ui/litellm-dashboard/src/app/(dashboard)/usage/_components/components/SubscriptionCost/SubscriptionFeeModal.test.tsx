import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import SubscriptionFeeModal from "./SubscriptionFeeModal";
import type { SubscriptionAccountUsage } from "./types";

const account: SubscriptionAccountUsage = {
  custom_llm_provider: "chatgpt",
  account_id: "acct_123",
  deployments: ["gpt-5.6-terra"],
  fee: null,
  billing_periods: [],
  fixed_cost: null,
};

const fill = (label: string, value: string) => fireEvent.change(screen.getByLabelText(label), { target: { value } });

describe("SubscriptionFeeModal", () => {
  it("submits normalized values for a new fee", async () => {
    const onSubmit = vi.fn().mockResolvedValue(undefined);
    render(<SubscriptionFeeModal visible account={account} onCancel={vi.fn()} onSubmit={onSubmit} />);

    expect(screen.getByText("Set monthly fee")).toBeInTheDocument();
    expect(screen.getByText("chatgpt account acct_123")).toBeInTheDocument();
    fill("Monthly fee", "25");
    fill("Currency", "usd");
    fill("Billing period start", "2026-10-01");
    fill("Label (optional)", "Team workspace");
    await userEvent.setup().click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1));
    const normalized = {
      monthly_fee: 25,
      currency: "USD",
      billing_period_start: "2026-10-01",
      label: "Team workspace",
    };
    expect(onSubmit).toHaveBeenCalledWith(normalized);
  });

  it("rejects a zero fee and a two-letter currency", async () => {
    const onSubmit = vi.fn().mockResolvedValue(undefined);
    const user = userEvent.setup();
    render(<SubscriptionFeeModal visible account={account} onCancel={vi.fn()} onSubmit={onSubmit} />);

    fill("Monthly fee", "0");
    fill("Billing period start", "2026-10-01");
    await user.click(screen.getByRole("button", { name: "Save" }));
    expect(await screen.findByText("Monthly fee must be greater than 0")).toBeInTheDocument();

    fill("Monthly fee", "25");
    fill("Currency", "US");
    await user.click(screen.getByRole("button", { name: "Save" }));
    expect(await screen.findByText("Currency must be a three-letter code")).toBeInTheDocument();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("prefills the existing fee when editing", () => {
    const existing: SubscriptionAccountUsage = {
      ...account,
      fee: {
        subscription_account_id: "fee-1",
        custom_llm_provider: "chatgpt",
        account_id: "acct_123",
        label: "Team workspace",
        monthly_fee: 25,
        currency: "EUR",
        billing_period_start: "2026-09-15",
        created_at: "2026-10-06T00:00:00Z",
        updated_at: "2026-10-06T00:00:00Z",
        created_by: "admin",
        updated_by: "admin",
      },
    };
    render(<SubscriptionFeeModal visible account={existing} onCancel={vi.fn()} onSubmit={vi.fn()} />);

    expect(screen.getByText("Edit monthly fee")).toBeInTheDocument();
    expect(screen.getByLabelText("Monthly fee")).toHaveValue(25);
    expect(screen.getByLabelText("Currency")).toHaveValue("EUR");
    expect(screen.getByLabelText("Billing period start")).toHaveValue("2026-09-15");
    expect(screen.getByLabelText("Label (optional)")).toHaveValue("Team workspace");
  });
});
