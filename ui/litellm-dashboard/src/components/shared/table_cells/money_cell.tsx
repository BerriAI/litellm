"use client";

import { formatNumberWithCommas, getSpendString } from "@/utils/dataUtils";

interface MoneyCellProps {
  value: number | null | undefined;
  decimals?: number;
  emptyText?: string;
  showZero?: boolean;
}

export function MoneyCell({ value, decimals = 4, emptyText = "-", showZero = false }: MoneyCellProps) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return <span className="text-muted-foreground">{emptyText}</span>;
  }
  if (value === 0) {
    if (!showZero) {
      return <span className="text-muted-foreground">-</span>;
    }
    return <span className="whitespace-nowrap">{`$${formatNumberWithCommas(0, decimals, false, true)}`}</span>;
  }
  return <span className="whitespace-nowrap">{getSpendString(value, decimals)}</span>;
}
