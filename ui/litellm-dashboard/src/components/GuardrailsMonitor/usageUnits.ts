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

export interface CounterMath {
  readonly counter: string;
  readonly units: number;
  readonly unpriced: number;
  readonly cost: number | null;
}

export const pricedUnits = ({ units, unpriced }: Pick<CounterMath, "units" | "unpriced">): number =>
  Math.max(units - unpriced, 0);

export const unitPrice = (row: CounterMath): number | null => {
  const priced = pricedUnits(row);
  return row.cost != null && priced > 0 ? row.cost / priced : null;
};

export const formatUnitPrice = (price: number): string => `$${price.toFixed(6).replace(/\.?0+$/, "")}`;

export const counterMathLine = (row: CounterMath): string => {
  const label = counterLabel(row.counter);
  const price = unitPrice(row);
  if (price == null) {
    return `${label}: ${row.units.toLocaleString()} ${row.units === 1 ? "unit" : "units"} with no known price, left out`;
  }
  const line = `${label}: ${pricedUnits(row).toLocaleString()} × ${formatUnitPrice(price)} = ${formatCost(row.cost)}`;
  return row.unpriced > 0 ? `${line} (${row.unpriced.toLocaleString()} unpriced left out)` : line;
};

export const unitsSumLine = (units: UsageUnits): string =>
  `${Object.entries(units)
    .map(([counter, n]) => `${counterLabel(counter)} ${n.toLocaleString()}`)
    .join(" + ")} = ${totalUnits(units).toLocaleString()}`;
