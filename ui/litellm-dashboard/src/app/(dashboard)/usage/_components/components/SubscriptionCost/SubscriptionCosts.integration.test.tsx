import { fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders, testQueryClient } from "@/../tests/test-utils";
import { subscriptionAccountCreateCall, subscriptionUsageCall } from "@/components/networking";
import SubscriptionCosts from "./SubscriptionCosts";
import type { SubscriptionUsageResponse } from "./types";

vi.mock("@/components/networking", () => ({
  formatDate: (date: Date) => date.toISOString().slice(0, 10),
  subscriptionUsageCall: vi.fn(),
  subscriptionAccountCreateCall: vi.fn().mockResolvedValue({}),
  subscriptionAccountUpdateCall: vi.fn().mockResolvedValue({}),
  subscriptionAccountDeleteCall: vi.fn().mockResolvedValue({}),
}));
vi.mock("@/lib/toast", () => ({ toast: { success: vi.fn(), fromError: vi.fn() } }));

const untracked: SubscriptionUsageResponse = {
  start_date: "2026-10-01",
  end_date: "2026-10-07",
  subscription_providers: ["chatgpt"],
  accounts: [
    {
      custom_llm_provider: "chatgpt",
      account_id: "acct_123",
      deployments: ["gpt-5.6-sol", "gpt-5.6-terra"],
      fee: null,
      billing_periods: [],
      fixed_cost: null,
    },
  ],
};

const fill = (label: string, value: string) => fireEvent.change(screen.getByLabelText(label), { target: { value } });

describe("SubscriptionCosts", () => {
  beforeEach(() => {
    testQueryClient.clear();
    vi.clearAllMocks();
    vi.mocked(subscriptionUsageCall).mockResolvedValue(untracked);
  });

  it("saves a fee set from the card on that card's account and reloads the card", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <SubscriptionCosts
        accessToken="sk-admin"
        startTime={new Date("2026-10-01T00:00:00Z")}
        endTime={new Date("2026-10-07T00:00:00Z")}
        enabled
        meteredSpend={0}
        canEdit
      />,
    );

    await user.click(await screen.findByRole("button", { name: "Set monthly fee" }));
    expect(screen.getByText("chatgpt account acct_123")).toBeInTheDocument();
    fill("Monthly fee", "25");
    fill("Currency", "USD");
    fill("Billing period start", "2026-10-01");
    await user.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => expect(subscriptionUsageCall).toHaveBeenCalledTimes(2));
    expect(subscriptionAccountCreateCall).toHaveBeenCalledWith(
      "sk-admin",
      expect.objectContaining({ custom_llm_provider: "chatgpt", account_id: "acct_123", monthly_fee: 25 }),
    );
  });
});
