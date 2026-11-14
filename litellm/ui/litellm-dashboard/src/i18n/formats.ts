import type { Formatter } from "i18next";

import { DEFAULT_LANGUAGE } from "./languages";

type DateInput = Date | number | string;
type InterpolationOptions = { readonly interpolationkey?: string };
type Build<Value, Options> = (lng: string, options: Options) => (value: Value) => string;

const cacheKey = (language: string, options: InterpolationOptions): string | null => {
  try {
    return (
      language + JSON.stringify(options, (name, option) => (name === options.interpolationkey ? undefined : option))
    );
  } catch {
    return null;
  }
};

export const cachedFormat = <Value, Options extends object>(build: Build<Value, Options>) => {
  const renders = new Map<string, (value: Value) => string>();
  return (value: Value, lng: string | undefined, options: Options & InterpolationOptions): string => {
    const language = (lng ?? DEFAULT_LANGUAGE).replaceAll("_", "-");
    const key = cacheKey(language, options);
    if (key === null) return build(language, options)(value);
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

const FORMATS = {
  number: cachedFormat((lng, options: Intl.NumberFormatOptions) => {
    const format = new Intl.NumberFormat(lng, { ...options, currency: USD.currency });
    return (value: number) => format.format(value);
  }),
  compact: cachedFormat((lng, options: Intl.NumberFormatOptions) => {
    const format = new Intl.NumberFormat(lng, { ...options, ...COMPACT, currency: USD.currency });
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
  date: dateFormat(DATE),
  monthday: dateFormat(MONTH_DAY),
  monthdaytime: dateFormat({ ...MONTH_DAY, ...CLOCK }),
  timestamp: dateFormat({ ...DATE, ...CLOCK }),
  numericdatetime: dateFormat(NUMERIC_DATE_TIME),
};

export const registerFormats = (formatter: Formatter): void =>
  Object.entries(FORMATS).forEach(([name, format]) => formatter.add(name, format));
