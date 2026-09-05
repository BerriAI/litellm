import { formatNumberWithCommas, getSpendString } from "@/utils/dataUtils";

export type UsageUnits = Readonly<Record<string, number>>;

export const formatCost = (cost: number | null | undefined): string => {
  if (cost == null) return "—";
  return cost === 0 ? `$${formatNumberWithCommas(0, 4)}` : getSpendString(cost, 4);
};

export const totalUnits = (units: UsageUnits): number => Object.values(units).reduce((sum, n) => sum + n, 0);

export const counterLabel = (counter: string): string =>
  counter
    .replace(/Units$/, "")
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .replace(/^./, (c) => c.toUpperCase());

export const unpricedSummary = (untracked: UsageUnits): string | null => {
  const total = totalUnits(untracked);
  return total > 0 ? `${total.toLocaleString()} ${total === 1 ? "unit" : "units"} unpriced` : null;
};
