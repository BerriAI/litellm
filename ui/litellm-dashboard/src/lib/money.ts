import type { components } from "@/lib/http/schema";

export type Currency = components["schemas"]["Currency"];

export const USD: Currency = { code: "USD", usd_rate: 1 };

export const fromUsd = (usd: number, currency: Currency): number => usd * currency.usd_rate;

export const toUsd = (amount: number, currency: Currency): number => amount / currency.usd_rate;

export type MoneyFormatOptions = {
  decimals?: number;
  abbreviate?: boolean;
  showZero?: boolean;
  showSmallAmount?: boolean;
};

function abbreviation(amount: number, enabled: boolean): { scale: number; suffix: string } {
  if (!enabled) return { scale: 1, suffix: "" };
  if (amount >= 1_000_000) return { scale: 1_000_000, suffix: "M" };
  if (amount >= 1_000) return { scale: 1_000, suffix: "K" };
  return { scale: 1, suffix: "" };
}

export function formatMoney(
  usd: number | null | undefined,
  currency: Currency,
  { decimals = 2, abbreviate = false, showZero = true, showSmallAmount = false }: MoneyFormatOptions = {},
): string {
  if (usd == null || !Number.isFinite(usd)) return "-";
  if (usd === 0 && !showZero) return "-";
  const amount = fromUsd(usd, currency);
  const { scale, suffix } = abbreviation(Math.abs(amount), abbreviate);
  const formatOptions: Intl.NumberFormatOptions = {
    style: "currency",
    currency: currency.code,
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  };
  const formatter = new Intl.NumberFormat("en-US", formatOptions);
  if (showSmallAmount && amount !== 0 && Number((amount / scale).toFixed(decimals)) === 0) {
    return `< ${formatter.format(1 / 10 ** decimals)}`;
  }
  return `${formatter.format(amount / scale)}${suffix}`;
}

export const formatUsd = (usd: number | null | undefined, options?: MoneyFormatOptions): string =>
  formatMoney(usd, USD, options);
