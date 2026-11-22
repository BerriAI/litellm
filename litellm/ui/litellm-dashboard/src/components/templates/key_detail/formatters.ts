const RELATIVE_STEPS: readonly (readonly [Intl.RelativeTimeFormatUnit, number])[] = [
  ["year", 365 * 24 * 60 * 60],
  ["month", 30 * 24 * 60 * 60],
  ["week", 7 * 24 * 60 * 60],
  ["day", 24 * 60 * 60],
  ["hour", 60 * 60],
  ["minute", 60],
];

const moneyFormat = (value: number): Intl.NumberFormatOptions => {
  if (value > 0 && value < 0.01) return { maximumSignificantDigits: 2 };
  return { minimumFractionDigits: 2, maximumFractionDigits: value > 0 && value < 1 ? 4 : 2 };
};

export const formatMoney = (value: number): string => `$${value.toLocaleString("en-US", moneyFormat(value))}`;

export const formatCount = (value: number): string => value.toLocaleString("en-US");

export const formatDate = (iso: string): string =>
  new Date(iso).toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric" });

const DATE_TIME_FORMAT: Intl.DateTimeFormatOptions = {
  year: "numeric",
  month: "short",
  day: "numeric",
  hour: "numeric",
  minute: "2-digit",
};

export const formatDateTime = (iso: string): string => new Date(iso).toLocaleString("en-US", DATE_TIME_FORMAT);

export const formatRelative = (iso: string, nowMs: number, locale: string): string => {
  const seconds = Math.round((Date.parse(iso) - nowMs) / 1000);
  const format = new Intl.RelativeTimeFormat(locale, { numeric: "auto" });
  const step = RELATIVE_STEPS.find(([, size]) => Math.abs(seconds) >= size);
  return step ? format.format(Math.round(seconds / step[1]), step[0]) : format.format(0, "minute");
};

export const middleTruncate = (value: string, keep = 4): string =>
  value.length > keep * 2 + 3 ? `${value.slice(0, keep)}\u2026${value.slice(-keep)}` : value;
