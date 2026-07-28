"use client";

import { formatNumberWithCommas, getSpendString } from "@/utils/dataUtils";

interface MoneyCellProps {
  value: number | null | undefined;
  decimals?: number;
  emptyText?: string;
  showZero?: boolean;
}

const placeholderClassName = "block w-full whitespace-nowrap text-right tabular-nums text-muted-foreground";

export function MoneyCell({ value, decimals = 4, emptyText = "-", showZero = false }: MoneyCellProps) {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return <span className={placeholderClassName}>{emptyText}</span>;
  }
  if (value === 0 && !showZero) {
    return <span className={placeholderClassName}>-</span>;
  }

  const formattedValue =
    value === 0 ? `$${formatNumberWithCommas(0, decimals, false, true)}` : getSpendString(value, decimals);
  const numericValue = formattedValue.replace("$", "").trim();

  return (
    <span
      data-slot="money-cell"
      className="flex w-full items-baseline justify-end gap-x-1 whitespace-nowrap text-right tabular-nums"
    >
      <span data-slot="money-cell-currency" aria-hidden="true">
        $
      </span>
      <span data-slot="money-cell-value" aria-hidden="true">
        {numericValue}
      </span>
      <span className="sr-only">{formattedValue}</span>
    </span>
  );
}
