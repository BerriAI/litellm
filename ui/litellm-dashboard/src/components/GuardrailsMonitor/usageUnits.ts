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

export const formatUnitPrice = (price: number): string => {
  const fixed = price.toFixed(6).replace(/\.?0+$/, "");
  return price > 0 && Number(fixed) === 0 ? "< $0.000001" : `$${fixed}`;
};

export interface MathRow {
  readonly label: string;
  readonly parts: readonly string[];
  readonly note: string | null;
}

export const counterMathRow = (row: CounterMath): MathRow => {
  const label = counterLabel(row.counter);
  const price = unitPrice(row);
  if (price == null) {
    return { label, parts: [row.units.toLocaleString(), "× —", "= —"], note: "no known price, left out" };
  }
  return {
    label,
    parts: [pricedUnits(row).toLocaleString(), `× ${formatUnitPrice(price)}`, `= ${formatCost(row.cost)}`],
    note:
      row.unpriced > 0
        ? `${row.unpriced.toLocaleString()} unpriced ${row.unpriced === 1 ? "unit" : "units"} left out`
        : null,
  };
};

export const unitsMathRows = (units: UsageUnits): readonly MathRow[] =>
  Object.entries(units).map(([counter, n]) => ({
    label: counterLabel(counter),
    parts: [n.toLocaleString()],
    note: null,
  }));

export const pricingIssueUrl = (unpriced: UsageUnits, provider?: string): string => {
  const subject = provider ? `${provider} guardrail` : "guardrail";
  const params = new URLSearchParams({
    template: "feature_request.yml",
    title: `[Feature]: add ${subject} pricing to the cost map`,
    "the-feature": `LiteLLM has no price for these ${subject} usage units, so the Guardrails Monitor leaves them out of the cost: ${Object.keys(unpriced).join(", ")}`,
  });
  return `https://github.com/BerriAI/litellm/issues/new?${params.toString()}`;
};
