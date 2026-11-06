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

interface DateCellProps {
  value: string | null | undefined;
  precision?: DatePrecision;
  fallback?: string;
}

const pad = (n: number): string => String(n).padStart(2, "0");

export const formatCellDate = (date: Date, precision: DatePrecision): string =>
  precision === "date"
    ? `${i18n.t(MONTHS[date.getMonth()])} ${date.getDate()}, ${date.getFullYear()}`
    : `${i18n.t(MONTHS[date.getMonth()])} ${date.getDate()}, ${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;

export const formatFullTimestamp = (date: Date): string => {
  const timeZone = Intl.DateTimeFormat().resolvedOptions().timeZone;
  const day = `${i18n.t(MONTHS[date.getMonth()])} ${date.getDate()}, ${date.getFullYear()}`;
  const time = `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
  return `${day}, ${time} (${timeZone})`;
};

export function DateCell({ value, precision = "datetime", fallback = "-" }: DateCellProps) {
  const date = value ? new Date(value) : null;
  if (!date || Number.isNaN(date.getTime())) {
    return <span className="text-muted-foreground">{fallback}</span>;
  }

  return (
    <CellTooltip
      content={formatFullTimestamp(date)}
      trigger={<span className="whitespace-nowrap">{formatCellDate(date, precision)}</span>}
    />
  );
}
