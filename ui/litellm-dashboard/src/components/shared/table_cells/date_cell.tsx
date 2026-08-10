"use client";

import { CellTooltip } from "./cell_tooltip";

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"] as const;

export type DatePrecision = "datetime" | "date";

interface DateCellProps {
  value: string | null | undefined;
  precision?: DatePrecision;
  fallback?: string;
  locale?: string;
}

const pad = (n: number): string => String(n).padStart(2, "0");

export const formatCellDate = (date: Date, precision: DatePrecision): string =>
  precision === "date"
    ? `${MONTHS[date.getMonth()]} ${date.getDate()}, ${date.getFullYear()}`
    : `${MONTHS[date.getMonth()]} ${date.getDate()}, ${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;

export const formatFullTimestamp = (date: Date): string => {
  const timeZone = Intl.DateTimeFormat().resolvedOptions().timeZone;
  const day = `${MONTHS[date.getMonth()]} ${date.getDate()}, ${date.getFullYear()}`;
  const time = `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
  return `${day}, ${time} (${timeZone})`;
};

const formatLocalizedDate = (date: Date, precision: DatePrecision, locale: string): string =>
  new Intl.DateTimeFormat(locale, {
    day: "numeric",
    month: "short",
    year: precision === "date" ? "numeric" : undefined,
    hour: precision === "datetime" ? "2-digit" : undefined,
    minute: precision === "datetime" ? "2-digit" : undefined,
    second: precision === "datetime" ? "2-digit" : undefined,
  }).format(date);

const formatLocalizedFullTimestamp = (date: Date, locale: string): string =>
  new Intl.DateTimeFormat(locale, {
    dateStyle: "medium",
    timeStyle: "long",
  }).format(date);

export function DateCell({ value, precision = "datetime", fallback = "-", locale }: DateCellProps) {
  const date = value ? new Date(value) : null;
  if (!date || Number.isNaN(date.getTime())) {
    return <span className="text-muted-foreground">{fallback}</span>;
  }

  return (
    <CellTooltip
      content={locale ? formatLocalizedFullTimestamp(date, locale) : formatFullTimestamp(date)}
      trigger={
        <span className="whitespace-nowrap">
          {locale ? formatLocalizedDate(date, precision, locale) : formatCellDate(date, precision)}
        </span>
      }
    />
  );
}
