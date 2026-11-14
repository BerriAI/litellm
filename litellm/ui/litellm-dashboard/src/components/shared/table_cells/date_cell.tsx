"use client";

import type { ParseKeys } from "i18next";

import i18n from "@/i18n/i18n";

import { CellTooltip } from "./cell_tooltip";

const MONTHS: readonly ParseKeys[] = [
  "shared.tableCells.dateCell.months.jan",
  "shared.tableCells.dateCell.months.feb",
  "shared.tableCells.dateCell.months.mar",
  "shared.tableCells.dateCell.months.apr",
  "shared.tableCells.dateCell.months.may",
  "shared.tableCells.dateCell.months.jun",
  "shared.tableCells.dateCell.months.jul",
  "shared.tableCells.dateCell.months.aug",
  "shared.tableCells.dateCell.months.sep",
  "shared.tableCells.dateCell.months.oct",
  "shared.tableCells.dateCell.months.nov",
  "shared.tableCells.dateCell.months.dec",
];

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

const monthName = ({ month }: DateParts): string => i18n.t(MONTHS[month]);

export const formatCellDate = (date: Date, precision: DatePrecision, timeZone: DateCellTimeZone = "local"): string => {
  const parts = dateParts(date, timeZone);
  const formatted =
    precision === "date"
      ? `${monthName(parts)} ${parts.day}, ${parts.year}`
      : `${monthName(parts)} ${parts.day}, ${clockTime(parts)}`;
  return timeZone === "UTC" ? `${formatted} UTC` : formatted;
};

export const formatFullTimestamp = (date: Date, timeZone: DateCellTimeZone = "local"): string => {
  const parts = dateParts(date, timeZone);
  const zoneName = timeZone === "UTC" ? "UTC" : Intl.DateTimeFormat().resolvedOptions().timeZone;
  return `${monthName(parts)} ${parts.day}, ${parts.year}, ${clockTime(parts)} (${zoneName})`;
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
