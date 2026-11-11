import type { DailyData } from "@/components/UsagePage/types";
import { addMetrics, type CustomerUsage, type Totals } from "../customers/_components/customerPortalData";

export type GroupBy = "customer" | "key" | "model";

export interface ChargebackRow extends Totals {
  customer: string;
  dimension: string | null;
}

const ZERO: Totals = { spend: 0, requests: 0, inputTokens: 0, outputTokens: 0 };

const hasTraffic = (r: Totals) => r.requests > 0 || r.spend > 0;
const bySpend = (a: Totals, b: Totals) => b.spend - a.spend;

export const rowsByCustomer = (usage: ReadonlyMap<string, CustomerUsage>): ChargebackRow[] =>
  [...usage.entries()]
    .map(([customer, u]) => ({
      customer,
      dimension: null,
      spend: u.spend,
      requests: u.requests,
      inputTokens: u.inputTokens,
      outputTokens: u.outputTokens,
    }))
    .filter(hasTraffic)
    .sort(bySpend);

export const rowsByCustomerAndKey = (usage: ReadonlyMap<string, CustomerUsage>): ChargebackRow[] =>
  [...usage.entries()]
    .flatMap(([customer, u]) => u.keys.map((k) => ({ ...k, customer, dimension: k.alias ?? k.hash.slice(0, 12) })))
    .filter(hasTraffic)
    .sort(bySpend);

export const rowsByCustomerAndModel = (perCustomer: ReadonlyMap<string, readonly DailyData[]>): ChargebackRow[] =>
  [...perCustomer.entries()]
    .flatMap(([customer, days]) => {
      const byModel = new Map<string, Totals>();
      for (const day of days) {
        for (const [model, m] of Object.entries(day.breakdown.models))
          byModel.set(model, addMetrics(byModel.get(model) ?? ZERO, m.metrics));
      }
      return [...byModel.entries()].map(([model, t]) => ({ ...t, customer, dimension: model }));
    })
    .filter(hasTraffic)
    .sort(bySpend);

export const sumRows = (rows: readonly Totals[]): Totals =>
  rows.reduce<Totals>(
    (acc, r) => ({
      spend: acc.spend + r.spend,
      requests: acc.requests + r.requests,
      inputTokens: acc.inputTokens + r.inputTokens,
      outputTokens: acc.outputTokens + r.outputTokens,
    }),
    ZERO,
  );

const FORMULA_PREFIX = /^[=+\-@\t\r]/;
const neutralise = (v: string) => (FORMULA_PREFIX.test(v) ? `'${v}` : v);
const csvCell = (v: string | number) => `"${(typeof v === "string" ? neutralise(v) : String(v)).replace(/"/g, '""')}"`;

export const toCsv = (
  rows: readonly ChargebackRow[],
  aliasOf: ReadonlyMap<string, string>,
  dimension: string | null,
): string => {
  const header = [
    "customer_id",
    "customer",
    ...(dimension ? [dimension] : []),
    "requests",
    "input_tokens",
    "output_tokens",
    "spend_usd",
  ];
  const lines = rows.map((r) => [
    r.customer,
    aliasOf.get(r.customer) ?? r.customer,
    ...(dimension ? [r.dimension ?? ""] : []),
    r.requests,
    r.inputTokens,
    r.outputTokens,
    r.spend.toFixed(6),
  ]);
  return [header, ...lines].map((l) => l.map(csvCell).join(",")).join("\n");
};
