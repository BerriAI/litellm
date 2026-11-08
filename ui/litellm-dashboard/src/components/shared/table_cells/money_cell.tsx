"use client";

import { formatNumberWithCommas, getSpendString } from "@/utils/dataUtils";

interface MoneyCellProps {
  value: number | null | undefined;
  decimals?: number;
  emptyText?: string;
  showZero?: boolean;
}

const placeholderClassName = "block w-full whitespace-nowrap text-right tabular-nums text-muted-foreground";
const moneyClassName = "block w-full whitespace-nowrap text-right tabular-nums";

export function MoneyCell({ value, decimals = 4, emptyText = "-", showZero = false }: MoneyCellProps) {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return <span className={placeholderClassName}>{emptyText}</span>;
  }
  if (value === 0 && !showZero) {
    return <span className={placeholderClassName}>-</span>;
  }

  const formattedValue =
    value === 0 ? `$${formatNumberWithCommas(0, decimals, false, true)}` : getSpendString(value, decimals);

  return (
    <span data-slot="money-cell" className={moneyClassName}>
      {formattedValue}
    </span>
  );
}
