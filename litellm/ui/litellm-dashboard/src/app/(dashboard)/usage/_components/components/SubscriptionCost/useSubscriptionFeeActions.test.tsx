import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { subscriptionAccountCreateCall, subscriptionAccountUpdateCall } from "@/components/networking";
import type { SubscriptionAccountUsage, SubscriptionFeeFormValues } from "./types";
import { useSubscriptionFeeActions } from "./useSubscriptionFeeActions";

vi.mock("@/components/networking", () => ({
  subscriptionAccountCreateCall: vi.fn().mockResolvedValue({}),
  subscriptionAccountUpdateCall: vi.fn().mockResolvedValue({}),
  subscriptionAccountDeleteCall: vi.fn(),
}));
vi.mock("@/lib/toast", () => ({ toast: { success: vi.fn(), fromError: vi.fn() } }));

const stored: SubscriptionFeeFormValues = {
  monthly_fee: 25,
  currency: "USD",
  billing_period_start: "2026-10-01",
  label: "Team plan (test value)",
};

const tracked: SubscriptionAccountUsage = {
  custom_llm_provider: "chatgpt",
  account_id: "acct_123",
  deployments: ["gpt-5.6-terra"],
  fee: {
    subscription_account_id: "sub-1",
    custom_llm_provider: "chatgpt",
    account_id: "acct_123",
    ...stored,
    created_at: "2026-10-01T00:00:00Z",
    updated_at: "2026-10-01T00:00:00Z",
    created_by: "admin",
    updated_by: "admin",
  },
  billing_periods: [{ start: "2026-10-01", end: "2026-10-31" }],
  fixed_cost: 25,
};

const submit = async (account: SubscriptionAccountUsage, values: SubscriptionFeeFormValues) => {
  const refetch = vi.fn().mockResolvedValue(undefined);
  const { result } = renderHook(() => useSubscriptionFeeActions({ accessToken: "sk-admin", refetch }));
  act(() => result.current.openFeeModal(account));
  await act(() => result.current.submitFee(values));
  return { refetch, result };
};

describe("useSubscriptionFeeActions", () => {
  beforeEach(() => vi.clearAllMocks());

  it("sends only the fields the admin changed when editing a fee", async () => {
    const { refetch, result } = await submit(tracked, { ...stored, monthly_fee: 30 });

    expect(subscriptionAccountUpdateCall).toHaveBeenCalledWith("sk-admin", {
      subscription_account_id: "sub-1",
      monthly_fee: 30,
    });
    expect(refetch).toHaveBeenCalledTimes(1);
    expect(result.current.feeModalAccount).toBeNull();
  });

  it("clears a label with an explicit null and nothing else", async () => {
    await submit(tracked, { ...stored, label: null });

    expect(subscriptionAccountUpdateCall).toHaveBeenCalledWith("sk-admin", {
      subscription_account_id: "sub-1",
      label: null,
    });
  });

  it("creates a fee with every field for an untracked account", async () => {
    const untracked: SubscriptionAccountUsage = { ...tracked, fee: null, billing_periods: [], fixed_cost: null };
    const createRequest = { custom_llm_provider: "chatgpt", account_id: "acct_123", ...stored };

    await submit(untracked, stored);

    expect(subscriptionAccountCreateCall).toHaveBeenCalledWith("sk-admin", createRequest);
    expect(subscriptionAccountUpdateCall).not.toHaveBeenCalled();
  });
});
