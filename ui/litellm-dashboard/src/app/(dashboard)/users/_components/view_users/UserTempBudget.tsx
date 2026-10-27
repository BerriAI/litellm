import React, { useState } from "react";
import { tempBudgetExpiryMs } from "@/components/shared/InheritedBudgetHint";
import { formatNumberWithCommas } from "@/utils/dataUtils";

interface TempBudget {
  increase: number;
  expiryMs: number;
}

interface UserTempBudgetProps {
  metadata: Record<string, unknown> | null;
}

const parseTempBudget = (metadata: Record<string, unknown> | null): TempBudget | null => {
  const increase = metadata?.temp_budget_increase;
  const expiry = metadata?.temp_budget_expiry;
  if (typeof increase !== "number" || typeof expiry !== "string") return null;
  const expiryMs = tempBudgetExpiryMs({ temp_budget_increase: increase, temp_budget_expiry: expiry });
  return expiryMs === null || Number.isNaN(expiryMs) ? null : { increase, expiryMs };
};

const useTempBudget = (metadata: Record<string, unknown> | null) => {
  const [nowMs] = useState(() => Date.now());
  const tempBudget = parseTempBudget(metadata);
  const isActive = tempBudget !== null && tempBudget.expiryMs > nowMs;
  return { tempBudget, isActive };
};

export function UserBudgetLimit({ maxBudget, metadata }: UserTempBudgetProps & { maxBudget: number | null }) {
  const { tempBudget, isActive } = useTempBudget(metadata);
  if (maxBudget === null) return <p>of Unlimited</p>;
  if (tempBudget === null || !isActive) return <p>of ${formatNumberWithCommas(maxBudget, 2)}</p>;
  return (
    <>
      <p>of ${formatNumberWithCommas(maxBudget + tempBudget.increase, 2)}</p>
      <p className="text-xs text-muted-foreground">
        Includes ${formatNumberWithCommas(tempBudget.increase, 2)} temporary increase until{" "}
        {new Date(tempBudget.expiryMs).toLocaleString()}
      </p>
    </>
  );
}

export function TempBudgetIncreaseDetail({ metadata }: UserTempBudgetProps) {
  const { tempBudget, isActive } = useTempBudget(metadata);
  if (tempBudget === null) return null;
  return (
    <div>
      <p className="font-medium">Temporary Budget Increase</p>
      <p>
        +${formatNumberWithCommas(tempBudget.increase, 4)} {isActive ? "until" : "expired"}{" "}
        {new Date(tempBudget.expiryMs).toLocaleString()}
      </p>
    </div>
  );
}
