import type { Formatter } from "i18next";

import { DEFAULT_LANGUAGE } from "./languages";

type DateInput = Date | number | string;
type InterpolationOptions = { readonly interpolationkey?: string };
type Build<Value, Options> = (lng: string, options: Options) => (value: Value) => string;

export const cachedFormat = <Value, Options extends object>(build: Build<Value, Options>) => {
  const renders = new Map<string, (value: Value) => string>();
  return (value: Value, lng: string | undefined, options: Options & InterpolationOptions): string => {
    const language = lng ?? DEFAULT_LANGUAGE;
    const key =
      language + JSON.stringify(options, (name, option) => (name === options.interpolationkey ? undefined : option));
    const existing = renders.get(key);
    if (existing) return existing(value);
    const render = build(language, options);
    renders.set(key, render);
    return render(value);
  };
};

const dateFormat = (preset: Intl.DateTimeFormatOptions) =>
  cachedFormat((lng, options: Intl.DateTimeFormatOptions) => {
    const format = new Intl.DateTimeFormat(lng, { ...options, ...preset });
    return (value: DateInput) => format.format(new Date(value));
  });

const USD = { style: "currency", currency: "USD", currencyDisplay: "symbol" } as const;
const COMPACT = { notation: "compact", maximumFractionDigits: 1 } as const;
const MONTH_DAY = { month: "short", day: "numeric" } as const;
const DATE = { ...MONTH_DAY, year: "numeric" } as const;
const CLOCK = { hour: "2-digit", minute: "2-digit", second: "2-digit", hourCycle: "h23" } as const;
const NUMERIC_DATE_TIME = {
  year: "numeric",
  month: "numeric",
  day: "numeric",
  hour: "numeric",
  minute: "2-digit",
  second: "2-digit",
} as const;

const DURATION_UNITS = [
  { unit: "day", size: 86_400_000, per: Infinity },
  { unit: "hour", size: 3_600_000, per: 24 },
  { unit: "minute", size: 60_000, per: 60 },
  { unit: "second", size: 1_000, per: 60 },
  { unit: "millisecond", size: 1, per: 1_000 },
] as const;

const duration = cachedFormat((lng: string) => {
  const units = DURATION_UNITS.map(({ unit, size, per }) => ({
    size,
    per,
    format: new Intl.NumberFormat(lng, { style: "unit", unit, unitDisplay: "narrow" }),
  }));
  const seconds = new Intl.NumberFormat(lng, { style: "unit", unit: "second", unitDisplay: "narrow" });
  const list = new Intl.ListFormat(lng, { type: "unit", style: "narrow" });
  return (milliseconds: number) => {
    const parts = units
      .map(({ size, per, format }) => ({ amount: Math.floor(milliseconds / size) % per, format }))
      .filter(({ amount }) => amount > 0)
      .map(({ amount, format }) => format.format(amount));
    return parts.length > 0 ? list.format(parts) : seconds.format(0);
  };
});

const FORMATS = {
  number: cachedFormat((lng, options: Intl.NumberFormatOptions) => {
    const format = new Intl.NumberFormat(lng, options);
    return (value: number) => format.format(value);
  }),
  compact: cachedFormat((lng, options: Intl.NumberFormatOptions) => {
    const format = new Intl.NumberFormat(lng, { ...options, ...COMPACT });
    return (value: number) => format.format(value);
  }),
  usd: cachedFormat((lng, options: Intl.NumberFormatOptions) => {
    const format = new Intl.NumberFormat(lng, { ...options, ...USD });
    return (value: number) => format.format(value);
  }),
  relativetime: cachedFormat(
    (lng, options: Intl.RelativeTimeFormatOptions & { range?: Intl.RelativeTimeFormatUnit }) => {
      const format = new Intl.RelativeTimeFormat(lng, options);
      return (value: number) => format.format(value, options.range ?? "day");
    },
  ),
  duration,
  date: dateFormat(DATE),
  monthday: dateFormat(MONTH_DAY),
  monthdaytime: dateFormat({ ...MONTH_DAY, ...CLOCK }),
  timestamp: dateFormat({ ...DATE, ...CLOCK }),
  numericdatetime: dateFormat(NUMERIC_DATE_TIME),
};

export const registerFormats = (formatter: Formatter): void =>
  Object.entries(FORMATS).forEach(([name, format]) => formatter.add(name, format));
