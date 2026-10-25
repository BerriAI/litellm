"use client";

import { CellTooltip } from "./cell_tooltip";
import { useLanguage } from "@/contexts/LanguageContext";
import type { Language } from "@/i18n/language";

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"] as const;

export type DatePrecision = "datetime" | "date";

interface DateCellProps {
  value: string | null | undefined;
  precision?: DatePrecision;
  fallback?: string;
}

const pad = (n: number): string => String(n).padStart(2, "0");

// 仅改变日期显示顺序，本地时间和时区沿用现有语义。
const chineseDate = (date: Date): string => `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;

export const formatCellDate = (date: Date, precision: DatePrecision, language: Language = "en"): string => {
  if (language === "zh-CN")
    return precision === "date"
      ? chineseDate(date)
      : `${chineseDate(date)} ${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
  return precision === "date"
    ? `${MONTHS[date.getMonth()]} ${date.getDate()}, ${date.getFullYear()}`
    : `${MONTHS[date.getMonth()]} ${date.getDate()}, ${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
};

export const formatFullTimestamp = (date: Date, language: Language = "en"): string => {
  const timeZone = Intl.DateTimeFormat().resolvedOptions().timeZone;
  const day =
    language === "zh-CN" ? chineseDate(date) : `${MONTHS[date.getMonth()]} ${date.getDate()}, ${date.getFullYear()}`;
  const time = `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
  return `${day}${language === "zh-CN" ? " " : ", "}${time} (${timeZone})`;
};

export function DateCell({ value, precision = "datetime", fallback = "-" }: DateCellProps) {
  const { language } = useLanguage();
  const date = value ? new Date(value) : null;
  if (!date || Number.isNaN(date.getTime())) {
    return <span className="text-muted-foreground">{fallback}</span>;
  }

  return (
    <CellTooltip
      content={formatFullTimestamp(date, language)}
      trigger={<span className="whitespace-nowrap">{formatCellDate(date, precision, language)}</span>}
    />
  );
}
