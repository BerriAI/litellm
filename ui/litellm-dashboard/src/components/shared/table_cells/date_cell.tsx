"use client";

import { CellTooltip } from "./cell_tooltip";

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"] as const;

export type DatePrecision = "datetime" | "date";

export type DateCellTimeZone = "local" | "UTC";

interface DateCellProps {
  value: string | null | undefined;
  precision?: DatePrecision;
  fallback?: string;
  timeZone?: DateCellTimeZone;
}

interface DateParts {
  year: number;
  month: number;
  day: number;
  hours: number;
  minutes: number;
  seconds: number;
}

const pad = (n: number): string => String(n).padStart(2, "0");

const dateParts = (date: Date, timeZone: DateCellTimeZone): DateParts =>
  timeZone === "UTC"
    ? {
        year: date.getUTCFullYear(),
        month: date.getUTCMonth(),
        day: date.getUTCDate(),
        hours: date.getUTCHours(),
        minutes: date.getUTCMinutes(),
        seconds: date.getUTCSeconds(),
      }
    : {
        year: date.getFullYear(),
        month: date.getMonth(),
        day: date.getDate(),
        hours: date.getHours(),
        minutes: date.getMinutes(),
        seconds: date.getSeconds(),
      };

const clockTime = ({ hours, minutes, seconds }: DateParts): string => `${pad(hours)}:${pad(minutes)}:${pad(seconds)}`;

export const formatCellDate = (date: Date, precision: DatePrecision, timeZone: DateCellTimeZone = "local"): string => {
  const parts = dateParts(date, timeZone);
  const formatted =
    precision === "date"
      ? `${MONTHS[parts.month]} ${parts.day}, ${parts.year}`
      : `${MONTHS[parts.month]} ${parts.day}, ${clockTime(parts)}`;
  return timeZone === "UTC" ? `${formatted} UTC` : formatted;
};

export const formatFullTimestamp = (date: Date, timeZone: DateCellTimeZone = "local"): string => {
  const parts = dateParts(date, timeZone);
  const zoneName = timeZone === "UTC" ? "UTC" : Intl.DateTimeFormat().resolvedOptions().timeZone;
  return `${MONTHS[parts.month]} ${parts.day}, ${parts.year}, ${clockTime(parts)} (${zoneName})`;
};

export function DateCell({ value, precision = "datetime", fallback = "-", timeZone = "local" }: DateCellProps) {
  const date = value ? new Date(value) : null;
  if (!date || Number.isNaN(date.getTime())) {
    return <span className="text-muted-foreground">{fallback}</span>;
  }

  return (
    <CellTooltip
      content={formatFullTimestamp(date, timeZone)}
      trigger={<span className="whitespace-nowrap">{formatCellDate(date, precision, timeZone)}</span>}
    />
  );
}
