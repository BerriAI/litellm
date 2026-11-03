import type { BillingPeriod } from "./types";

export const ATTRIBUTION_RULE =
  "A subscription fee is counted once for each monthly billing period whose start date falls inside the selected dates. Fixed costs sit beside metered spend and are never added to it, to key or team spend, or to budgets.";

export const formatFee = (amount: number, currency: string): string => `${amount.toFixed(2)} ${currency}`;

const parseLocalDay = (day: string): Date => {
  const [year, month, date] = day.split("-").map(Number);
  return new Date(year, month - 1, date);
};

export const formatBillingPeriod = ({ start, end }: BillingPeriod): string => {
  const startDate = parseLocalDay(start);
  const endDate = parseLocalDay(end);
  const startLabel = startDate.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: startDate.getFullYear() !== endDate.getFullYear() ? "numeric" : undefined,
  });
  const endLabel = endDate.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
  return `${startLabel} - ${endLabel}`;
};

export const fixedCostByCurrency = (
  accounts: readonly { fee: { currency: string } | null; fixed_cost: number | null }[],
): readonly { currency: string; total: number }[] => {
  const totals = new Map<string, number>();
  accounts.forEach((account) => {
    if (account.fee === null || account.fixed_cost === null) return;
    totals.set(account.fee.currency, (totals.get(account.fee.currency) ?? 0) + account.fixed_cost);
  });
  return Array.from(totals, ([currency, total]) => ({ currency, total }));
};
