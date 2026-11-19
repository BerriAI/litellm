import type { ReferenceExchangeRates } from "@/app/(dashboard)/hooks/referenceRates/useReferenceRates";

export const MAX_RATE_AGE_DAYS = 7;

const DAY_MS = 24 * 60 * 60 * 1000;

export type SpendEstimate = { readonly amount: number; readonly rateDate: string };

export const estimatedSpend = (
  usdSpend: number,
  rates: Pick<ReferenceExchangeRates, "usd_per_unit" | "rate_date">,
  currency: string,
  now: Date,
): SpendEstimate | null => {
  const rate = new Map(Object.entries(rates.usd_per_unit)).get(currency);
  const today = Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate());
  const ageDays = (today - Date.parse(`${rates.rate_date}T00:00:00Z`)) / DAY_MS;
  if (rate === undefined || !(rate > 0) || !(ageDays >= 0 && ageDays <= MAX_RATE_AGE_DAYS)) return null;
  return { amount: usdSpend / rate, rateDate: rates.rate_date };
};
