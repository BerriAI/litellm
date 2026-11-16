export const MAX_RATE_AGE_DAYS = 7;

const DAY_MS = 24 * 60 * 60 * 1000;

export type SpendEstimate = { readonly amount: number; readonly rateDate: string };

/** The USD spend at the current reference rate for `currency`, or null when that rate is missing or too old to use. */
export const estimatedSpend = (
  usdSpend: number,
  usdPerUnit: Readonly<Record<string, number>>,
  rateDate: string,
  currency: string,
  now: Date,
): SpendEstimate | null => {
  const rate = new Map(Object.entries(usdPerUnit)).get(currency);
  const today = Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate());
  const ageDays = (today - Date.parse(`${rateDate}T00:00:00Z`)) / DAY_MS;
  if (rate === undefined || !(rate > 0) || !(ageDays >= 0 && ageDays <= MAX_RATE_AGE_DAYS)) return null;
  return { amount: usdSpend / rate, rateDate };
};
